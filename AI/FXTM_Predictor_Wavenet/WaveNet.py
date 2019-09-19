from tensorflow.python.keras.layers import *
import tensorflow.python.keras as keras
import tensorflow.python.keras.losses as l
import os
from tensorflow.python.keras.callbacks import *
from tensorflow.python.keras.regularizers import *

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'


class WaveNet:
    def __init__(self, n_conv_filters, n_residual, n_skip, num_layer, n_repeat, filter_width=2, conditional=True,
                 l2_lambda=0.001):
        self.n_repeat = n_repeat
        self.dilation_rates = [2 ** i for i in range(num_layer)]
        self.preprocess_layer = Conv1D(n_residual, kernel_size=1)
        self.conv_layers = []
        self.pp_layers = []
        self.skip_layers = []
        self.cond_pp_layers = []
        self.fc = Conv1D(n_skip, kernel_size=1, activation='relu', kernel_regularizer=l2(l2_lambda))
        self.l2_lambda = l2_lambda
        self.conditional = conditional
        for i in range(n_repeat):
            for dilation_rate in self.dilation_rates:
                if conditional:
                    self.conv_layers.append(Conv1D(filters=n_conv_filters, kernel_size=filter_width,
                                                   padding='causal', dilation_rate=dilation_rate,
                                                   kernel_regularizer=l2(l2_lambda)))

                    self.cond_pp_layers.append(Dense(n_conv_filters, activation=None))

                else:
                    self.conv_layers.append(Conv1D(filters=n_conv_filters, kernel_size=filter_width,
                                                   padding='causal', dilation_rate=dilation_rate,
                                                   kernel_regularizer=l2(l2_lambda), activation='selu'
                                                   ))

                self.pp_layers.append(Dense(n_residual, activation=None))

                self.skip_layers.append(Dense(n_skip, kernel_regularizer=l2(l2_lambda),
                                              activation=None))

    def compile(self, input_dim, output_dim, optimizer, cond_latent_dim=0, loss='mae', metrics=None):
        self.enc_features = input_dim[1] * input_dim[2]
        self.dec_features = output_dim[1] * output_dim[2]

        projection = Conv1D(self.dec_features, kernel_size=1, activation=None,
                            kernel_regularizer=l2(self.l2_lambda))

        encoder_input = Input(shape=(None, input_dim[1], input_dim[2]), name="encoder_in")
        encoder_input_flat = Reshape(target_shape=(-1, self.enc_features), name="enc_reshape")(encoder_input)

        decoder_input = Input(shape=(None, output_dim[1], output_dim[2]), name="decoder_in")
        decoder_input_flat = Reshape(target_shape=(-1, self.dec_features), name="dec_reshape")(decoder_input)
        decoder_input_lagged = Lambda(lambda x: x[:, :-1], name="decoder_take_away_last")(decoder_input_flat)

        input = Concatenate(axis=1)([encoder_input_flat, decoder_input_lagged])
        cond_input = Input(shape=(cond_latent_dim,))

        if self.conditional:
            # broadcaster = Lambda(
            #     lambda x: K.repeat_elements(x, K.int_shape(input)[1], axis=1))
            # broadcaster = RepeatVector(K.shape(input)[1])# broadcast over time dim

            def feedforward(x, condition):
                skips = []
                x = self.preprocess_layer(x)

                for i in range(len(self.conv_layers)):
                    x_conv = self.conv_layers[i](x)

                    cond = self.cond_pp_layers[i](condition)
                    # cond = broadcaster(cond)
                    x_conv = Activation('selu')(Add()([x_conv, cond]))

                    z = self.pp_layers[i](x_conv)

                    x = Add()([x, z])

                    skips.append(z)

                out = Activation('selu')(Add()(skips))
                out = self.fc(out)
                out = projection(out)

                return out

            train_x = feedforward(input, cond_input)
        else:
            def feedforward(x):
                skips = []
                x = self.preprocess_layer(x)

                for i in range(len(self.conv_layers)):
                    x_conv = self.conv_layers[i](x)

                    z = self.pp_layers[i](x_conv)

                    x = Add()([x, z])

                    skip = self.skip_layers[i](x_conv)

                    skips.append(skip)

                out = Activation('selu')(Add()(skips))
                out = self.fc(out)
                out = projection(out)

                return out

            train_x = feedforward(input)

        pred = Lambda(lambda x: x[:, -output_dim[0]:, :], name="train_prediction")(train_x)
        pred = Reshape(target_shape=(output_dim[0], output_dim[1], output_dim[2]))(pred)

        if self.conditional:
            self.model = keras.models.Model([encoder_input, decoder_input, cond_input], pred)
            self.model.compile(optimizer=optimizer, loss=loss, metrics=metrics)
        else:
            self.model = keras.models.Model([encoder_input, decoder_input], pred)
            self.model.compile(optimizer=optimizer, loss=loss, metrics=metrics)

        self.model.summary()

    def predict(self, input, how_many, condition=None):
        # assuming input = (1, input_seq, symbols, features)
        # condition = (1, latent_dim)
        if self.conditional:
            inp = np.copy(input)
            pred = np.zeros_like(inp)
            pred = pred[:, :how_many]

            for i in range(how_many):
                encX = inp[:, :-1]
                decX = np.concatenate([inp[:, -1:], np.zeros_like(inp)[:, 0:1]], axis=1)
                last_pred = self.model.predict([encX, decX, condition])[:, -1, :, :]  # (1(batch), 1(sym), 1(feat))
                pred[:, i] = last_pred
                inp = np.concatenate([inp, pred[:, i:i + 1]], axis=1)

            return pred
        else:
            inp = np.copy(input)
            pred = np.zeros_like(inp)
            pred = pred[:, :how_many]

            for i in range(how_many):
                encX = inp[:, :-1]
                decX = np.concatenate([inp[:, -1:], np.zeros_like(inp)[:, 0:1]], axis=1)
                last_pred = self.model.predict([encX, decX])[:, -1, :, :]  # (1(batch), 1(sym), 1(feat))
                pred[:, i] = last_pred
                inp = np.concatenate([inp, pred[:, i:i + 1]], axis=1)

            return pred
