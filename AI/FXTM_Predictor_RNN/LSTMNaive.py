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
import tensorflow as tf
from tensorflow.python.keras.optimizers import Adam
from tensorflow.python.keras.losses import huber_loss
from tensorflow.python.keras.metrics import mean_absolute_percentage_error

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
os.chdir("D:/Projects/AI")

def mape(true, pred):
    return mean_absolute_percentage_error(true, pred)

def diff_mda(true, pred):
    res = tf.equal(tf.sign(true), tf.sign(pred))
    return tf.reduce_mean(tf.cast(res, tf.float32))


def visualize(gen, model, p):
    sp = 60
    fig = plt.figure()
    ax = fig.add_subplot(1, 1, 1)

    def animate(i):
        testX, testY = gen.get_test_data()
        sampleX = testX[i + sp: i + 1 + sp]
        pred = model.predict(sampleX, p)

        sampleX = sampleX[0]
        pred = pred[0]

        sampleY = testY[i + sp:i + sp + 1]
        sampleY = sampleY[0]

        true = np.concatenate([sampleX[-p - 4:], sampleY[:p + 4]], axis=0)
        pred = np.concatenate([sampleX[-p - 4:], pred], axis=0)

        ax.clear()

        ax.plot(true[:, 0, 0:1])
        ax.plot(pred[:, 0, 0:1])

    anim = animation.FuncAnimation(fig, animate, frames=3000, interval=1000)
    plt.show()

def visualize2(gen, model):
    sp = 60
    fig = plt.figure()
    ax = fig.add_subplot(1, 1, 1)

    def animate(i):
        testX, testY = gen.get_test_data()
        sampleX = testX[sp:sp+64]
        sampleX = sampleX[:, -i-1:]
        pred = model.predict(sampleX, 1)

        sampleY = testY[sp:sp+64]

        true = np.reshape(sampleY, [sampleY.shape[0]*sampleY.shape[1], sampleY.shape[2], sampleY.shape[3]])
        pred = np.reshape(pred, [pred.shape[0]*pred.shape[1], pred.shape[2], pred.shape[3]])
        ax.clear()
        # print(true.shape)
        # ax.axvline(x=p - 1)
        # ax.axhline(y=0)
        ax.plot(true[:, 0, 0:1])
        ax.plot(pred[:, 0, 0:1])

    anim = animation.FuncAnimation(fig, animate, frames=3000, interval=600)
    plt.show()


class NaiveResidualLSTM:
    def __init__(self, layers, n_pp):
        self.preprocess_layer = Dense(n_pp, activation=None)
        self.LSTMs = [CuDNNLSTM(layer, return_sequences=True) for layer in layers]
        self.postprocess_layers = [Dense(n_pp, activation=None) for i in range(len(layers))]

    def compile(self, input_dim, output_dim, optimizer=Adam(decay=0.1), loss=huber_loss):
        input = Input(shape=(None, input_dim[1], input_dim[2]))
        f_input = Reshape(target_shape=(-1, input_dim[1] * input_dim[2]))(input)

        x = self.preprocess_layer(f_input)

        for i in range(len(self.LSTMs)):
            out = self.LSTMs[i](x)
            out = self.postprocess_layers[i](out)
            x = Add()([x, out])

        x = Dense(output_dim[1] * output_dim[2], activation=None)(x)  # projection to output dimension

        pred = Lambda(lambda x: x[:, -output_dim[0]:, :], name="train_prediction")(x)
        pred = Reshape((output_dim[0], output_dim[1], output_dim[2]))(pred)

        self.model = keras.models.Model(input, pred)
        self.model.compile(optimizer=optimizer, loss=loss, metrics=[diff_mda])
        self.model.summary()

    def predict(self, x, pred_length):  # x.shape = (1, timestep, symbol, hlc)
        preds = []
        for i in range(pred_length):
            prediction = self.model.predict(x)  # (1, 1, symbol, hlc)
            preds.append(prediction)
            x = np.concatenate([x, prediction], axis=1)
        preds = np.concatenate(preds, axis=1)
        return preds


if __name__ == "__main__":
    train_ratio = 0.8
    seq_length = 32
    output_count = 1
    output_count_for_test = 1
    pred_length = 1
    symbols = ["EURUSD"]

    layers = [32, 32, 32, 32, 32, 32]
    n_pp = 32

    batch_size = 512
    epochs = 500

    mfile = './SavedModel/RNN/NaiveResidualLSTM.h5'
    model_saver = ModelCheckpoint(mfile, save_best_only=True, save_weights_only=True)

    gen = VaeGen(train_ratio, seq_length, output_count, symbols,
                 ohlc_list=["close", "high", "low"],
                 n_ema=None,
                 diff=True,
                 logtransform=False,
                 window_step=1,
                 test_output_count=output_count_for_test,
                 last_date=datetime(2019, 7, 28), num_samples=92160,
                 timeframe=MetaTrader5.MT5_TIMEFRAME_M1)

    trainX, trainY = gen.get_train_data()
    validX, validY = gen.get_val_data()

    print(trainY.shape)

    input_dim = gen.input_dim
    output_dim = gen.output_dim

    test = NaiveResidualLSTM(layers, n_pp)
    test.compile(input_dim, output_dim)
    test.model.fit(trainX, trainY, batch_size=batch_size, epochs=epochs, callbacks=[model_saver],
                   validation_data=(validX, validY))
    test.model.load_weights(mfile)

    visualize2(gen, test)
