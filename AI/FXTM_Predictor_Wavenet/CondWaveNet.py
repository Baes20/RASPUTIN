from tensorflow.python.keras.layers import *
import tensorflow.python.keras as keras
import tensorflow as tf
from tensorflow.python.keras.callbacks import *
from tensorflow.python.keras.regularizers import *

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'


def PermaDropout(rate):
    return Lambda(lambda x: K.dropout(x, level=rate))


class CondWaveNet:
    def __init__(self, n_conv_filters, n_pp, n_fc, num_layer, n_repeat, filter_width=2, l2_lambda=0.001):
        self.conv_filters = n_conv_filters
        self.filter_width = filter_width
        self.n_repeat = n_repeat
        self.n_pp = n_pp
        self.dilation_rates = [2 ** i for i in range(num_layer)]
        self.conv_layers = []
        self.postprocess = []
        self.skip_layers = []
        self.fc = Dense(n_fc, activation='selu')
        self.main_parametrization = Dense(n_pp, activation=None,
                                          name="main_param",
                                          )
        self.l2_lambda = l2_lambda
        count = 0
        for i in range(n_repeat):
            for dilation_rate in self.dilation_rates:
                self.conv_layers.append(Conv1D(filters=n_conv_filters, kernel_size=filter_width,
                                               padding='causal', dilation_rate=dilation_rate,
                                               name="dilated_convolution_" + str(count), activation='selu',
                                              ))

                self.postprocess.append(Dense(n_pp, activation=None))

                count += 1

    def compile(self, input_dim, output_dim, optimizer, main_input_index=0, default_loss='mse', metrics=None):
        # assumed shape = (timestep, n) where input[...,0] is a main prediction
        # mode=0: traditional, else: train enc_inp as well
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.enc_features = input_dim[1] * input_dim[2]
        self.dec_features = output_dim[1] * output_dim[2]

        # input = concat(encoder_seq + decoder_seq[:,-1,:]
        encoder_input = Input(shape=(None, input_dim[1], input_dim[2]), name="encoder_in")
        encoder_input_flat = Reshape(target_shape=(-1, self.enc_features), name="enc_reshape")(encoder_input)

        decoder_input = Input(shape=(None, output_dim[1], output_dim[2]), name="decoder_in")
        decoder_input_flat = Reshape(target_shape=(-1, self.dec_features), name="dec_reshape")(decoder_input)
        decoder_input_lagged = Lambda(lambda x: x[:, :-1, :], name="decoder_take_away_last")(decoder_input_flat)

        train_main = tf.unstack(Concatenate(axis=1)([encoder_input_flat, decoder_input_lagged]), axis=-1)
        input_train = tf.stack(train_main[main_input_index:main_input_index + 1], axis=-1)
        cond_train = tf.stack(train_main[:main_input_index] + train_main[main_input_index + 1:], axis=-1)

        projection = Dense(1)

        cond_parametrization = [Dense(self.n_pp, activation=None,
                                      name="cond_param_" + str(i),
                                      ) for i in range(self.enc_features - 1)]

        cond_conv = [Conv1D(filters=self.conv_filters, kernel_size=self.filter_width,
                            padding='causal', dilation_rate=1,
                            name="cond_convolution_" + str(i), activation='selu',
                            ) for i in range(self.enc_features - 1)]

        cond_postprocess = [Dense(self.n_pp, activation=None,
                                  name="cond_pp_" + str(i),
                                  ) for i in range(self.enc_features - 1)]

        def feedforward(x, h):
            hs = tf.unstack(h, axis=-1)  # [(batch_size, timestep),...]
            hs = [tf.expand_dims(h_, axis=-1) for h_ in hs]  # [(batch_size, timestep, 1),...]
            out = x
            skips = []
            for i in range(len(self.conv_layers)):

                if i == 0:
                    x_c = self.conv_layers[i](out)
                    x_c = self.postprocess[i](x_c)
                    x_paramed = self.main_parametrization(out)
                    x_out = Add()([x_c, x_paramed])

                    h_out = []

                    for j in range(len(hs)):
                        h_c = cond_conv[j](hs[j])
                        h_c = cond_postprocess[j](h_c)
                        h_paramed = cond_parametrization[j](hs[j])
                        h_out.append(Add()([h_c, h_paramed]))

                    out = Add()([x_out] + h_out)
                    skips.append(out)

                else:
                    out_c = self.conv_layers[i](out)
                    out_c = self.postprocess[i](out_c)
                    out = Add()([out, out_c])
                    skips.append(out_c)

            out = Activation('relu')(Add()(skips))
            out = self.fc(out)
            out = projection(out)
            return out

        train_x = feedforward(input_train, cond_train)
        # infer_x = feedforward(input_infer, cond_infer)

        train_pred = Lambda(lambda x: x[:, -output_dim[0]:, :], name="train_prediction")(train_x)
        # infer_pred = Lambda(lambda x: x[:, -output_dim[0]:, :], name="train_prediction")(infer_x)

        train_pred = Reshape(target_shape=(output_dim[0], 1, 1))(train_pred)
        # infer_pred = Reshape(target_shape=(output_dim[0], 1, 1))(infer_pred)

        self.model_train = keras.models.Model([encoder_input, decoder_input], train_pred)
        # self.model_infer = keras.models.Model([encoder_input, decoder_input], infer_pred)

        self.model_train.compile(optimizer=optimizer, loss=default_loss, metrics=metrics)
        self.model_train.summary()

    def predict(self, input):
        inp = np.copy(input)
        encX = inp[:, :-1]
        decX = np.concatenate([inp[:, -1:], np.zeros_like(inp)[:, 0:1]], axis=1)
        last_pred = self.model_train.predict([encX, decX])[:, -1:, :, :]  # (1(batch), 1(sym), 1(feat))

        return last_pred
