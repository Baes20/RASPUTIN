from tensorflow.python.keras.layers import *
import tensorflow.python.keras as keras
import os
from tensorflow.python.keras.callbacks import ModelCheckpoint, TensorBoard
from DataCookers.VAEdataset import VaeGen
import MetaTrader5
from datetime import datetime
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import animation
from rwa import RWACell
import tensorflow.python.keras.backend as K
from DataCookers.FXTMCategorical import FXTMCategorical

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
os.chdir("D:/Projects/AI")


def predict(model, data, how_many):  # data = (1, timestep, symbol, ohlc)
    preds = []
    for i in range(how_many):
        pred = model.predict(data)
        preds.append(pred)
        data = np.concatenate([data, pred], axis=1)
        data = data[:, 1:]
    return np.concatenate(preds, axis=1)


def visualize(gen, model, p):
    sp = 0
    fig = plt.figure()
    ax = fig.add_subplot(1, 1, 1)
    testX, testY = gen.get_test_data()
    past_window = 4

    def animate(i):
        sampleX = testX[i + sp: i + 1 + sp]
        pred = predict(model, sampleX, p)

        sampleX = np.squeeze(sampleX)
        pred = np.squeeze(pred)
        sampleY = testY[i + sp:i + sp + 1]
        sampleY = np.squeeze(sampleY)

        true = np.concatenate([sampleX[-p - past_window:], sampleY[:p]], axis=0)
        pred = np.concatenate([sampleX[-p - past_window:], pred], axis=0)

        true = gen.stdizer.inverse_transform(true)
        pred = gen.stdizer.inverse_transform(pred)
        # true = np.cumsum(true, axis=0)
        # pred = np.cumsum(pred, axis=0)

        ax.clear()

        ax.axvline(x=p + past_window - 1)
        ax.plot(true[:])
        ax.plot(pred[:])

    anim = animation.FuncAnimation(fig, animate, frames=3000, interval=300)
    plt.show()


class RecurrentWeightedAverage(keras.layers.RNN):
    def __init__(self, n_units, return_sequences=False, return_state=False, go_backwards=False, stateful=False,
                 unroll=False, **kwargs):
        self.cell = RWACell(n_units)
        self.units = n_units
        super(RecurrentWeightedAverage, self).__init__(self.cell,
                                                       return_sequences=return_sequences,
                                                       return_state=return_state,
                                                       go_backwards=go_backwards,
                                                       stateful=stateful,
                                                       unroll=unroll,
                                                       **kwargs)

    def get_initial_state(self, inputs):
        initial_state = K.zeros_like(inputs)
        initial_state = K.sum(initial_state, axis=(1, 2))
        initial_state = K.expand_dims(initial_state)
        initial_state = K.tile(initial_state, [1, self.units])  # (samples, output_dim)
        n = K.identity(initial_state)
        d = K.identity(initial_state)
        h = K.identity(initial_state)

        dtype = initial_state.dtype.name
        min_value = np.array([1E38]).astype(dtype).item()
        a_max = K.identity(initial_state) - min_value
        h = h + self.cell.recurrent_activation(K.expand_dims(self.cell.initial_attention, axis=0))

        return [n, d, h, a_max]


if __name__ == "__main__":
    train_ratio = 0.8
    seq_length = 256
    output_count = 1
    output_count_for_test = 8
    pred_length = 4
    symbols = ["EURUSD"]
    ohlc_list = ["close"]
    ema = None
    oscillator = None
    n_samples = 92160
    diff = False
    timeframe = MetaTrader5.MT5_TIMEFRAME_D1
    layers = [50]
    n_bins = 64
    batch_size = 512
    epochs = 100

    mfile = './SavedModel/RNN/RWA.h5'
    tfile = './Summary/RNN/RWA/'
    model_saver = ModelCheckpoint(mfile, save_best_only=True, save_weights_only=True)
    tensorboard = TensorBoard(log_dir=tfile, write_graph=True, histogram_freq=1)
    gen = VaeGen(train_ratio, seq_length, output_count, symbols, ohlc_list, datetime(2019, 6, 18), n_ema=ema,
                 osc_list=oscillator,
                 test_output_count=pred_length,
                 window_step=1,
                 num_samples=n_samples,
                 diff=diff,
                 timeframe=timeframe)

    trainX, trainY = gen.get_train_data()
    validX, validY = gen.get_val_data()
    input_dim = gen.input_dim
    output_dim = gen.output_dim

    # test = keras.models.Sequential()
    # test.add(Input(shape=(None, input_dim[1])))
    # test.add(RecurrentWeightedAverage(layers[0]))
    # test.add(Dense(output_dim[1]))
    # test.add(Reshape(target_shape=(1, output_dim[1])))
    # test.compile(optimizer='adam', loss='mae')
    # test.summary()
    # # test.fit(trainX, trainY, batch_size=batch_size, epochs=epochs, callbacks=[model_saver],
    # #                validation_data=(validX, validY))
    # test.load_weights(mfile)
    #
    # visualize(gen, test, pred_length)
