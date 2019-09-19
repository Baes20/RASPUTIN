from AI.FXTM_Predictor_Wavenet.WaveNet import WaveNet
from tensorflow.python.keras.callbacks import *
from tensorflow.python.keras.optimizers import Adam
from DataCookers.FXTMCategorical import FXTMCategorical
import os
from datetime import datetime
import MetaTrader5
import matplotlib.pyplot as plt
from matplotlib import animation
from AI.FXTM_Predictor_Wavenet.WaveNetCategorical import CatWN

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
os.chdir("D:/Projects/AI")


def mda(actual: np.ndarray, predicted: np.ndarray):
    """ Mean Directional Accuracy """
    return np.mean((np.sign(actual[1:] - actual[:-1]) == np.sign(predicted[1:] - predicted[:-1])).astype(int))


def visualize(gen, model, pred_length=4):
    sp = 0
    fig = plt.figure()
    ax = fig.add_subplot(2, 1, 1)
    ax2 = fig.add_subplot(2, 1, 2)
    metric = []
    testX, testY = gen.get_test_data()

    def animate(i):
        ax.clear()
        ax2.clear()
        sampleX = testX[i + sp: i + 1 + sp]
        pred, pred_prob = model.predict(sampleX, pred_length)

        sampleX = gen.inverse_digitize(sampleX[0])
        pred = gen.inverse_digitize(pred[0])

        sampleY = testY[i + sp:i + sp + 1]

        sampleY = gen.inverse_digitize(sampleY[0])

        metric.append((sampleY[:pred_length] - pred) / (sampleY[:pred_length] + 0.000001))
        # samplecond = cond[i + sp: i + sp + 1]

        # C = rawX[-pred_length - 1]

        true = np.concatenate([sampleX[-pred_length-4:], sampleY[:pred_length]], axis=0)
        pred = np.concatenate([sampleX[-pred_length-4:], pred], axis=0)

        # true = np.cumsum(true, axis=0)
        # pred = np.cumsum(pred, axis=0)

        ax.plot(true)
        ax.plot(pred)

        m = np.stack(metric, axis=0)
        m = np.transpose(m, (0, 2, 1))
        m = np.reshape(m, (m.shape[0]*m.shape[1], m.shape[2]))
        # ax2.plot(np.squeeze(pred_prob))
        ax2.hist(m, range=(-3, 3), bins=60)

    anim = animation.FuncAnimation(fig, animate, frames=3000, interval=10)
    plt.show()


########################CommonParams##############################

train_ratio = 0.8
symbol = ["EURUSD"]
ohlc = ["close"]
ema = None
oscillator = None
ema_list = None
diff = True
predict_count = 1
date = datetime(2019, 6, 14)
n_sample = 92160
n_bins = 64
timeframe = MetaTrader5.MT5_TIMEFRAME_H1

##################################################################

#######################WaveNetHyperParams#########################

seq_length = 1024
output_count = 32
batch_size = 16
epoch = 60
l2_lambda = 0.0001

n_residual = 32
n_filter = 32
n_skip = 64
n_layer = 10
n_repeat = 1
kernel_size = 2

gen = FXTMCategorical(train_ratio, seq_length, output_count, symbol, ohlc, date, n_ema=ema, osc_list=oscillator,
                      test_output_count=predict_count,
                      window_step=output_count,
                      num_samples=n_sample,
                      diff=diff,
                      num_bins=n_bins,
                      timeframe=timeframe)
plt.plot(gen.dataset)
gen.digitize()
print(gen.dataset.shape)
plt.plot(gen.inverse_digitize(gen.dataset))
plt.show()
trainX, trainY = gen.get_train_data()
validX, validY = gen.get_val_data()

print(trainX.shape)

print(gen.dataset.shape)
Hparam = '-' + str(n_filter) + '-' + str(n_residual) + '-' + str(n_skip) + '-' + str(n_layer) + '-' + str(n_repeat)

mfile = './SavedModel/CatWaveNet/CatWaveNet' + Hparam + '.h5'

model_saver = ModelCheckpoint(mfile, save_best_only=True, save_weights_only=True)

##################################################################

wavenet = CatWN(n_filter, n_residual, n_skip, n_layer, n_repeat, filter_width=kernel_size, conditional=False,
                l2_lambda=l2_lambda)

wavenet.compile(gen.get_input_dim(), gen.get_output_dim(), optimizer='adam')
# wavenet.model.fit([trainX, trainY], trainY,
#                   validation_data=([validX, validY], validY),
#                   batch_size=batch_size, epochs=epoch, callbacks=[model_saver])
wavenet.model.load_weights(mfile)

visualize(gen, wavenet, pred_length=predict_count)

