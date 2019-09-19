from tensorflow.python.keras.layers import *
import tensorflow.python.keras as keras
import os
from tensorflow.python.keras.callbacks import ModelCheckpoint
from DataCookers.VAEdataset import VaeGen
import MetaTrader5
from datetime import datetime
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import animation

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'


def visualize(gen, model, p):
    sp = 60
    fig = plt.figure()
    ax = fig.add_subplot(1, 1, 1)

    def animate(i):
        sampleX = gen.testX[i + sp: i + 1 + sp]
        pred = model.predict(sampleX, p)

        sampleX = np.squeeze(sampleX)
        pred = np.squeeze(pred)

        sampleY = gen.testY[i + sp:i + sp + 1]
        sampleY = np.squeeze(sampleY)

        true = np.concatenate([sampleX[-p:], sampleY[:p]], axis=0)
        pred = np.concatenate([sampleX[-p:], pred], axis=0)

        # true = np.cumsum(true, axis=0)
        # pred = np.cumsum(pred, axis=0)

        ax.clear()

        ax.axvline(x=p - 1)
        ax.plot(true[:, 0, -1])
        ax.plot(pred[:, 0, -1])

    anim = animation.FuncAnimation(fig, animate, frames=3000, interval=1000)
    plt.show()


class Seq2Seq:
    def __init__(self, layers, n_pp):
        self.layers = layers

        self.enc_preprocess_layer = Dense(n_pp, activation=None)
        self.dec_preprocess_layer = Dense(n_pp, activation=None)

        self.encoder_LSTMlayers = [CuDNNLSTM(layer, return_state=True, return_sequences=True) for layer in layers]
        self.encoder_pp_layers = [Dense(n_pp, activation=None) for i in range(len(layers))]

        self.decoder_LSTMlayers = [CuDNNLSTM(layer, return_sequences=True) for layer in layers]
        self.decoder_pp_layers = [Dense(n_pp, activation=None) for i in range(len(layers))]

    def encode(self, x):
        x = self.enc_preprocess_layer(x)
        enc_states = []
        for i in range(len(self.encoder_LSTMlayers)):  # encoder
            out, h, c = self.encoder_LSTMlayers[i](x)
            out = self.encoder_pp_layers[i](out)
            x = Add()([x, out])
            state = [h, c]
            enc_states.append(state)

        return enc_states

    def decode(self, x, states):
        x = self.dec_preprocess_layer(x)
        for i in range(len(self.decoder_LSTMlayers)):
            out = self.decoder_LSTMlayers[i](x, initial_state=states[i])
            out = self.decoder_pp_layers[i](out)
            x = Add()([x, out])
        return x

    def compile(self, input_dim, output_dim, optimizer='adam', loss='mae'):
        enc_input = Input(shape=(None, input_dim[1], input_dim[2]))
        f_enc_input = Reshape(target_shape=(-1, input_dim[1] * input_dim[2]))(enc_input)

        dec_input = Input(shape=(None, output_dim[1], output_dim[2]))
        f_dec_input = Reshape(target_shape=(-1, output_dim[1] * output_dim[2]))(dec_input)
        dec_input_train = Lambda(lambda x: x[:, :-1], name="decoder_take_away_last")(f_dec_input)
        enc_last = Lambda(lambda x: x[:, -1:])(f_enc_input)
        dec_input_train = Concatenate(axis=1)([enc_last, dec_input_train])

        self.projection_layer = Dense(output_dim[1] * output_dim[2], activation=None)

        enc_states = self.encode(f_enc_input)

        dec_out = self.decode(dec_input_train, enc_states)

        dec_out = self.projection_layer(dec_out)  # projection to output dimension

        pred = Reshape((-1, output_dim[1], output_dim[2]))(dec_out)

        self.train_model = keras.models.Model([enc_input, dec_input], pred)
        self.train_model.compile(optimizer=optimizer, loss=loss)

        self.encoder = keras.models.Model(enc_input, enc_states)

        dec_states = [[Input(shape=(layer,)), Input(shape=(layer,))] for layer in self.layers]
        dec_infer = Input(shape=(None, output_dim[1], output_dim[2]))
        dec_infer_f = Reshape(target_shape=(-1, output_dim[1] * output_dim[2]))(dec_infer)

        pred_infer = self.decode(dec_infer_f, dec_states)
        pred_infer = self.projection_layer(pred_infer)
        pred_infer = Reshape(target_shape=(-1, output_dim[1], output_dim[2]))(pred_infer)

        self.decoder = keras.models.Model([dec_infer, dec_states], pred_infer)

    def predict(self, x, pred_length):  # x.shape = (1, timestep, symbol, hlc)
        enc_states = self.encoder.predict(x)
        dec_in = x[:, -1:]
        preds = []
        for i in range(pred_length):
            prediction = self.decoder.predict([dec_in] + enc_states)  # (1, 1, symbol, hlc)
            preds.append(prediction)
            dec_in = prediction
        preds = np.concatenate(preds, axis=1)
        return preds


if __name__ == "__main__":
    train_ratio = 0.8
    seq_length = 128
    output_count = 32
    output_count_for_test = 32
    pred_length = 4
    symbols = ["EURUSD", "GBPUSD", "EURGBP"]

    layers = [50, 50]
    n_pp = 100

    batch_size = 128
    epochs = 100

    mfile = './SavedModel/RNN/Seq2SeqLSTM.h5'
    model_saver = ModelCheckpoint(mfile, save_best_only=True, save_weights_only=True)

    gen = VaeGen(train_ratio, seq_length, output_count, symbols, test_output_count=pred_length,
                 last_date=datetime(2019, 5, 28), num_samples=92160,
                 timeframe=MetaTrader5.MT5_TIMEFRAME_M1)

    test = Seq2Seq(layers, n_pp)
    test.compile(gen.input_dim, gen.output_dim, optimizer='adam', loss='mae')
    # test.train_model.fit([gen.trainX, gen.trainY], gen.trainY, batch_size=batch_size, epochs=epochs, callbacks=[model_saver],
    #                validation_data=([gen.validX, gen.validY], gen.validY))
    test.train_model.load_weights(mfile)

    visualize(gen, test, pred_length)
