import os
from AI.FXTM_Predictor_Wavenet.WaveEnsemble import CondWaveEnsemble
from DataCookers.FXTMdatasetCond import CondMarketDataGenerator
from datetime import datetime
import MetaTrader5
import matplotlib.pyplot as plt
from matplotlib import animation
import numpy as np
import time

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
plt.rcParams['animation.ffmpeg_path'] = 'C:/Users/Borealis/Downloads/ffmpeg-20190517-96c79d4-win64-static/bin/ffmpeg'

# tensorboard --host 127.0.0.1 --logdir=D:\Projects\AI\Summary\CondWavenet\

#######################HyperParams#########################
train_ratio = 0.8
seq_length = 768
output_count = 256
batch_size = 64
predict_count = 8

dropout_rate = 0.1
n_filter = 32
n_fc = 16
n_layer = 9
n_repeat = 1
filter_width = 3
l2_lambda = 0.0001
epoch = 20

symbol = ["EURUSD", "EURGBP", "GBPUSD"]

gen = CondMarketDataGenerator(train_ratio, seq_length, output_count, symbol,
                              datetime(2019, 5, 10), 80000,
                              timeframe=MetaTrader5.MT5_TIMEFRAME_M15)

###########################################################

pyoneer = CondWaveEnsemble(gen, symbol, n_filter, n_fc, n_layer, n_repeat, filter_width, l2_lambda, dropout_rate)

# pyoneer.train(batch_size, epoch, optim=Adam(lr=0.001), loss=mean_absolute_error)



def MASE(testing_series, prediction_series):
    n = testing_series.shape[0]
    d = np.abs(np.diff(testing_series)).sum() / (n - 1)

    errors = np.abs(testing_series - prediction_series)
    return errors.mean() / d

MASEs = []
sp = 500
fig = plt.figure()
ax = fig.add_subplot(1, 1, 1)
# ax2 = fig.add_subplot(2, 1, 2)
def animate(i):
    sampleX = gen.validX[i + sp:i + sp + 1]
    sampleY = gen.validY[i + sp:i + sp + 1]
    rawX = gen.rawX[i + sp:i + sp + 1]
    rawY = gen.rawY[i + sp:i + sp + 1]
    firstX = gen.nodifX[i + sp:i + sp + 1, 0]
    firstX_ema = gen.nodif_emaX[i + sp:i + sp + 1, 0]
    # firstY = gen.nodifY[i + 1000:i + 1000 + 1, 0]

    start = time.time()
    pred = pyoneer.predict(sampleX, predict_count)
    elapsed_time = time.time() - start

    sampleX = np.squeeze(sampleX)
    sampleY = np.squeeze(sampleY)
    pred = np.squeeze(pred)
    rawX = np.squeeze(rawX)
    rawY = np.squeeze(rawY)
    sampleX = gen.deStandardize(gen.prev_dataset, sampleX)
    sampleY = gen.deStandardize(gen.prev_dataset, sampleY)
    pred = gen.deStandardize(gen.prev_dataset, pred)

    # MASEs.append(np.mean([MASE(sampleY[:predict_count, i], pred[:, i]) for i in range(3)]))

    # MASE_mean = sum(MASEs)/len(MASEs)

    true = np.cumsum(np.concatenate([sampleX[-predict_count:], sampleY[:predict_count]], axis=0), axis=0)
    pred = np.cumsum(np.concatenate([sampleX[-predict_count:], pred[:]], axis=0), axis=0)
    raw = np.cumsum(np.concatenate([rawX[-predict_count:], rawY[:predict_count]], axis=0), axis=0)
    firstX = np.squeeze(firstX)
    firstX_ema = np.squeeze(firstX_ema)
    true = np.squeeze(true) + firstX_ema
    pred = np.squeeze(pred) + firstX_ema
    raw = np.squeeze(raw) + firstX

    constraint = (1 / pred[:, 0]) * (pred[:, 1]) * (pred[:, 2])
    mean = np.mean(constraint[:40], axis=0)

    ax.clear()
    # ax2.clear()
    # ax2.plot(np.diff(true[:, 0], axis=0))
    # ax2.plot(np.diff(pred[:, 0], axis=0))
    # ax2.set_ylim(0.999, 1.001)
    # ax2.plot(constraint)
    # ax2.axvline(x=39)
    # ax2.axhline(y=mean+0.0001)
    # ax2.axhline(y=mean-0.0001)
    ax.axvline(x=predict_count - 1)
    ax.plot(true[:, 0])
    ax.plot(pred[:, 0])
    # ax.plot(raw[:, 0])

# def animate(i):
#     sampleX = gen.validX[i + sp:i + sp + 1]
#     sampleY = gen.validY[i + sp:i + sp + 1]
#     rawX = gen.rawX[i + sp:i + sp + 1]
#     rawY = gen.rawY[i + sp:i + sp + 1]
#
#     start = time.time()
#     pred = pyoneer.predict(sampleX, predict_count)
#     elapsed_time = time.time() - start
#
#     sampleX = np.squeeze(sampleX)
#     sampleY = np.squeeze(sampleY)
#     pred = np.squeeze(pred)
#     sampleX = gen.deStandardize(gen.prev_dataset, sampleX)
#     sampleY = gen.deStandardize(gen.prev_dataset, sampleY)
#     pred = gen.deStandardize(gen.prev_dataset, pred)
#
#     # MASEs.append(np.mean([MASE(sampleY[:predict_count, i], pred[:, i]) for i in range(3)]))
#
#     # MASE_mean = sum(MASEs)/len(MASEs)
#
#     true = np.concatenate([sampleX[-predict_count:], sampleY[:predict_count]], axis=0)
#     pred = np.concatenate([sampleX[-predict_count:], pred[:]], axis=0)
#     # raw = np.concatenate([rawX[-predict_count:], rawY[:predict_count]], axis=0)
#     constraint = (1 / pred[:, 0]) * (pred[:, 1]) * (pred[:, 2])
#     mean = np.mean(constraint[:40], axis=0)
#
#     ax.clear()
#     ax2.clear()
#     ax2.plot(np.diff(true[:, 0], axis=0))
#     ax2.plot(np.diff(pred[:, 0], axis=0))
#     # ax2.set_ylim(0.999, 1.001)
#     # ax2.plot(constraint)
#     # ax2.axvline(x=39)
#     # ax2.axhline(y=mean+0.0001)
#     # ax2.axhline(y=mean-0.0001)
#     ax.axvline(x=predict_count - 1)
#     ax.plot(true[:, 0])
#     ax.plot(pred[:, 0])
#     # ax.plot(raw[:, 0])


anim = animation.FuncAnimation(fig, animate, frames=300, interval=500)

# FFwriter = animation.FFMpegWriter(fps=10, extra_args=['-vcodec', 'libx264'])
# anim.save('./basic_animation.mp4', writer=FFwriter)

plt.show()
