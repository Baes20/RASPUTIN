from tensorflow.python.keras.layers import *
import tensorflow.python.keras as keras
import os
from tensorflow.python.keras.callbacks import ModelCheckpoint
from tensorflow.python.keras.regularizers import *
from DataCookers.VAEdataset import VaeGen
import MetaTrader5
from datetime import datetime
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import animation
import tensorflow as tf
from tensorflow.python.keras.optimizers import Adam

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
os.chdir("D:/Projects/AI")

def diff_mda(true, pred):
    res = tf.equal(tf.sign(true), tf.sign(pred))
    return tf.reduce_mean(tf.cast(res, tf.float32))

def visualize(gen, model, p):
    sp = 60
    fig = plt.figure()
    ax = fig.add_subplot(2, 1, 1)
    ax2 = fig.add_subplot(2, 1, 2)
    def animate(i):
        testX, testY = gen.get_test_data()
        sampleX = testX[i + sp: i + 1 + sp]
        pred, score = model.predict(sampleX, p)

        sampleX = sampleX[0]
        pred = pred[0]
        score = score[0]

        sampleY = testY[i + sp:i + sp + 1]
        sampleY = sampleY[0]

        true = np.concatenate([sampleX[-p-4:], sampleY[:p+4]], axis=0)
        pred = np.concatenate([sampleX[-p-4:], pred], axis=0)

        # true = np.cumsum(true, axis=0)
        # pred = np.cumsum(pred, axis=0)

        ax.clear()
        ax2.clear()

        ax.axvline(x=p - 1)
        ax.plot(true[:, 0, 0:1])
        ax.plot(pred[:, 0, 0:1])
        y = np.squeeze(score)
        x = range(len(y))
        ax2.bar(x, y)
    anim = animation.FuncAnimation(fig, animate, frames=3000, interval=1000)
    plt.show()



class AttnLSTM:
    def __init__(self, layers, n_pp):
        self.preprocess_layer = Dense(n_pp, activation=None)
        self.LSTMs = [CuDNNLSTM(layer, return_sequences=True) for layer in layers]
        self.postprocess_layers = [Dense(n_pp, activation=None) for i in range(len(layers))]
        self.n_pp = n_pp

    def compile(self, input_dim, output_dim, optimizer=Adam(), loss='mae'):
        input = Input(shape=(None, input_dim[1], input_dim[2]))
        f_input = Reshape(target_shape=(-1, input_dim[1] * input_dim[2]))(input)

        x = self.preprocess_layer(f_input)

        for i in range(len(self.LSTMs)):
            out = self.LSTMs[i](x)
            out = self.postprocess_layers[i](out)
            x = Add()([x, out])

        # attention starts
        pred = Lambda(lambda x: x[:, -output_dim[0]:, :])(x)  # None, 1, 50
        residual = Lambda(lambda x: x[:, :-output_dim[0], :])(x)
        pred_T = Permute((2, 1))(pred)  # None, 50, 1
        a = Dot(axes=(2, 1))([residual, pred_T]) # None, None, 1
        score = Softmax(axis=1)(a) #None, None, 1
        context = Multiply()([residual, score]) # None, None, 50
        context = Lambda(lambda x: K.sum(x, axis=1))(context) #None, 50
        context = Reshape(target_shape=(K.shape(pred)[1], self.n_pp))(context)
        x = Concatenate(axis=-1)([pred, context]) #None, 1, 100
        x = Dense(self.n_pp, activation='relu')(x) # relu is the best so far
        # attention ends

        x = Dense(output_dim[1] * output_dim[2], activation=None)(x)  # projection to output dimension

        pred = Reshape((output_dim[0], output_dim[1], output_dim[2]))(x)

        self.model = keras.models.Model(input, [pred, score])
        # self.model.compile(optimizer=optimizer, loss=loss, metrics=[diff_mda])
        self.model.summary()

        self.model_train = keras.models.Model(input, pred)
        self.model_train.compile(optimizer=optimizer, loss=loss, metrics=[diff_mda])

    def predict(self, x, pred_length):  # x.shape = (1, timestep, symbol, hlc)
        preds = []
        attns = []
        for i in range(pred_length):
            prediction, attn_score = self.model.predict(x)  # (1, 1, symbol, hlc)
            preds.append(prediction)
            attns.append(attn_score)
            x = np.concatenate([x, prediction], axis=1)
        preds = np.concatenate(preds, axis=1)
        # attns = np.mean(attns, axis=-1)
        return preds, attns


if __name__ == "__main__":
    train_ratio = 0.8
    seq_length = 32
    output_count = 1
    output_count_for_test = 5
    pred_length = 1
    symbols = ["EURUSD"]

    layers = [64]
    n_pp = 32

    batch_size = 512
    epochs = 100

    mfile = './SavedModel/RNN/LSTMAttn.h5'
    model_saver = ModelCheckpoint(mfile, save_best_only=True, save_weights_only=True)

    gen = VaeGen(train_ratio, seq_length, output_count, symbols,
                 ohlc_list=["close", "high", "low"],
                 n_ema=None,
                 diff=True,
                 logtransform=False,
                 test_output_count=output_count_for_test,
                 last_date=datetime(2019, 7, 28), num_samples=92160,
                 timeframe=MetaTrader5.MT5_TIMEFRAME_M1)

    trainX, trainY = gen.get_train_data()
    validX, validY = gen.get_val_data()

    test = AttnLSTM(layers, n_pp)
    test.compile(gen.input_dim, gen.output_dim, optimizer=Adam(decay=0.15))
    test.model_train.fit(trainX, trainY, batch_size=batch_size, epochs=epochs, callbacks=[model_saver],
                   validation_data=(validX, validY))
    test.model.load_weights(mfile)

    visualize(gen, test, pred_length)

