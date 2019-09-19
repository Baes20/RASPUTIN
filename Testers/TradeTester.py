from DataCookers.TradeSim import FXTMTradeSimulator
from AI.FXTM_Predictor_Wavenet.WaveNet import WaveNet
from datetime import datetime
import MetaTrader5
import matplotlib.pyplot as plt
import numpy as np
import os
from DataCookers.VAEdataset import VaeGen
import random
from matplotlib import animation

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
os.chdir("D:/Projects/AI")

timeframe = MetaTrader5.MT5_TIMEFRAME_M15
date = datetime(2019, 8, 9)
test_date = datetime(2019, 1, 1)
symbol = ["EURUSD"]
#######################WaveNetHyperParams#########################

seq_length = 33
output_count = 1
l2_lambda_W = 0.000000000000001

n_residual = 128
n_filter = 128
n_skip = 128
n_layer = 5
n_repeat = 1
kernel_size = 2

# gen_W = VaeGen(train_ratio, seq_length_W, output_count_W, symbol, ohlc, date, n_ema=ema, osc_list=oscillator,
#                test_output_count=predict_count,
#                window_step=output_count_W,
#                num_samples=n_sample,
#                diff=diff,
#                timeframe=timeframe)

# trainX, trainY = gen_W.get_train_data()


Hparam = '-' + str(n_filter) + '-' + str(n_residual) + '-' + str(n_skip) + '-' + str(n_layer) + '-' + str(
    n_repeat) + '-' + str(timeframe)

mfile_W = './SavedModel/WaveNet/WaveNet' + Hparam + '.h5'

##################################################################

sim = FXTMTradeSimulator(seq_length, symbol=symbol, date=date, n_sample=92160, timeframe=timeframe)

gen_test = VaeGen(0.8, seq_length, 1, symbol, ["close", "high", "low"], date, n_ema=None,
                  ema_list=None,
                  test_output_count=1,
                  window_step=1,
                  num_samples=20000,
                  diff=True,
                  logtransform=False,
                  preprocess=True,
                  timeframe=timeframe)

stdizer = gen_test.stdizer

# initialize simulator

wavenet = WaveNet(n_filter, n_residual, n_skip, n_layer, n_repeat, filter_width=kernel_size, conditional=False,
                  l2_lambda=l2_lambda_W)

wavenet.compile(gen_test.input_dim, gen_test.output_dim, optimizer='adam')

wavenet.model.load_weights(mfile_W)
del gen_test

log = []

fig = plt.figure()
ax = fig.add_subplot(2, 1, 1)
ax2 = fig.add_subplot(2, 1, 2)

def animate(i):
    current_candle = sim.get_current_candle()
    X = sim.getX()  # (33, 1, 3)
    X = np.diff(X, axis=0)  # (32, 1, 3)
    X = stdizer.transform(np.squeeze(X))  # (32, 3)
    pred_return = wavenet.predict(np.expand_dims(np.expand_dims(X, axis=0), axis=2), 1)  # (1, 1, 1, 3)
    pred_return = stdizer.inverse_transform(np.expand_dims(np.squeeze(pred_return), axis=0))
    pred_return = np.squeeze(pred_return)  # [close, high, low]
    # prediction part over

    rand = random.uniform(0, 1)
    if pred_return[1] > 0.0001 and pred_return[2] > 0.0001:
        # if high change is more than 0.0005
        sim.buy_entry(1)  # buy 1 lot

    plt_X = sim.getX()[-16:, 0]  # (16, 3)
    plt_pred = plt_X[-1:] + np.expand_dims(pred_return, axis=0)  # (1, 3)
    plt_Y = sim.getY()[0]  # (1, 3)

    ax.clear()
    ax2.clear()
    ax.plot(np.concatenate([plt_X, plt_Y], axis=0)[:, 1:])  # only show high and low
    ax.plot(np.concatenate([plt_X, plt_pred], axis=0)[:, 1:])  # only show high and low
    ax2.plot(log)
    sim.update(exit_profit_threshold=(current_candle[1] - current_candle[0]))  # once it reaches last high, just sell
    sim.summary()
    log.append(sim.getBalance())

anim = animation.FuncAnimation(fig, animate, frames=3000, interval=1)
plt.show()


# plt.plot(log)
# plt.show()
