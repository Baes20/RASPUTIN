from tensorflow.python.keras.callbacks import *
from tensorflow.python.keras.optimizers import Adam
from tensorflow.python.keras.losses import *
import tensorflow as tf
import os
from datetime import datetime
import MetaTrader5
from AI.FXTM_Predictor_Wavenet.CondWaveNet import CondWaveNet
import matplotlib.pyplot as plt
from DataCookers.VAEdataset import VaeGen
from matplotlib import animation

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
os.chdir("D:/Projects/AI")


def mda(true, pred):
    res = tf.equal(tf.sign(true[1:] - true[:-1]), tf.sign(pred[1:] - pred[:-1]))
    return tf.reduce_mean(tf.cast(res, tf.float32))


def diff_mda(true, pred):
    res = tf.equal(tf.sign(true), tf.sign(pred))
    return tf.reduce_mean(tf.cast(res, tf.float32))


def visualize(gen, model):
    sp = 2
    fig = plt.figure()
    ax = fig.add_subplot(1, 1, 1)
    # ax2 = fig.add_subplot(2, 1, 2)
    testX, testY = gen.get_test_data()
    past_window = 16

    def animate(i):
        sampleX = testX[i + sp: i + 1 + sp]
        sampleY = testY[i + sp:i + sp + 1]

        pred = model.predict(sampleX)
        pred = pred[0, :, 0, :]  # (timestep, 1)
        sampleY = np.reshape(sampleY, (predict_count, gen.input_dim[1] * gen.input_dim[2]))[:, 0:1] # (timestep, 1)
        sampleX = np.reshape(sampleX, (gen.input_dim[0], gen.input_dim[1] * gen.input_dim[2]))[-past_window:, 0:1]

        sampleX = gen.output_stdizer.inverse_transform(sampleX)
        sampleY = gen.output_stdizer.inverse_transform(sampleY)
        pred = gen.output_stdizer.inverse_transform(pred)

        ax.clear()
        # ax2.clear()

        ax.plot(np.concatenate([sampleX, sampleY], axis=0))
        ax.plot(np.concatenate([sampleX, pred], axis=0))
        # ax2.plot((np.abs(sampleY[:, 0] - pred_1[:, 0])/sampleY[:, 0])*100)

    anim = animation.FuncAnimation(fig, animate, frames=3000, interval=600)
    # anim2 = animation.FuncAnimation(fig2, animate, frames=3000, interval=600)
    # anim3 = animation.FuncAnimation(fig3, animate, frames=3000, interval=600)

    plt.show()


#######################HyperParams#########################
seq_length = 512
output_count = 32
batch_size = 16

n_filter = 32
n_pp = 32
n_fc = 64
n_layer = 8
n_repeat = 1
filter_width = 3
l2_lambda = 0.00001
epoch = 80

Hparam = '-' + str(n_pp) + '-' + str(n_filter) + '-' + str(n_fc) + '-' + str(n_layer) + '-' + str(
    n_repeat) + '-' + str(filter_width) + '-' + str(l2_lambda)
###########################################################

########################CommonParams##############################

train_ratio = 0.8
symbol = [
    "EURUSD", "EURGBP", "GBPUSD", "USDCHF", "USDJPY", "USDCAD", "NZDUSD", "CHFJPY", "EURAUD", "AUDJPY", "GBPCAD",
    "GBPJPY", "EURCAD"]
# symbol = ["EURUSD", "EURGBP", "GBPUSD"]
ohlc = ["close"]
ema = 5
oscillator = None
ema_list = None
diff = False
predict_count = 1
date = datetime(2019, 6, 24)
n_sample = 80000
timeframe = MetaTrader5.MT5_TIMEFRAME_M1

##################################################################

mfile = './SavedModel/WaveNet/CondWaveNet_16' + Hparam + '.h5'
vae_mfile = '.\SavedModel\VAE/VAE.h5'
summarydir = './Summary/CondWavenet/' + Hparam + '/'
tensorboard = TensorBoard(log_dir=summarydir, write_graph=True, histogram_freq=1, write_images=True)
model_saver = ModelCheckpoint(mfile, save_best_only=True, save_weights_only=True)

# tensorboard --host 127.0.0.1 --logdir=D:\Projects\AI\Summary\CondWavenet\

gen = VaeGen(train_ratio, seq_length, output_count, symbol, ohlc, date, n_ema=ema, ema_list=ema_list,
             osc_list=oscillator,
             test_output_count=predict_count,
             window_step=output_count,
             num_samples=n_sample,
             diff=diff,
             timeframe=timeframe)

trainX, trainY = gen.get_train_data()
validX, validY = gen.get_val_data()

print(validY[..., 0:1, 0:1].shape)

# test = WaveNetMK0(n_filter, n_fc, n_layer)
test = CondWaveNet(n_filter, n_pp, n_fc, n_layer, n_repeat, filter_width=filter_width, l2_lambda=l2_lambda)
test.compile(gen.input_dim, gen.output_dim, optimizer=Adam(lr=0.001), default_loss=mean_absolute_error,
             main_input_index=gen.get_index_from_dict["EURUSD_close"], metrics=[mean_absolute_percentage_error])
test.model_train.fit([trainX, trainY], trainY[:, :, 0:1, 0:1], batch_size=batch_size, epochs=epoch,
                     callbacks=[model_saver],
                     validation_data=([validX, validY], validY[:, :, 0:1, 0:1]))
test.model_train.load_weights(mfile)
visualize(gen, test)
