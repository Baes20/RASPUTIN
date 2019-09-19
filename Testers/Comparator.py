from AI.FXTM_Predictor_RNN.LSTMNaive import NaiveResidualLSTM
from AI.FXTM_Predictor_Wavenet.WaveNet import WaveNet
from tensorflow.python.keras.callbacks import *
import os
from datetime import datetime
import MetaTrader5
import matplotlib.pyplot as plt
from matplotlib import animation
from DataCookers.VAEdataset import VaeGen
import tensorflow as tf
from mpl_toolkits.mplot3d import Axes3D
from tensorflow.python.keras.losses import *
from tensorflow.python.keras.optimizers import *

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
os.chdir("D:/Projects/AI")


def diff_mda(true, pred):
    res = tf.equal(tf.sign(true), tf.sign(pred))
    return tf.reduce_mean(tf.cast(res, tf.float32))


def diff_mda2(true, pred):
    res = np.equal(np.sign(true), np.sign(pred))
    return res


def mda(true, pred):
    res = tf.equal(tf.sign(true[1:] - true[:-1]), tf.sign(pred[1:] - pred[:-1]))
    return tf.reduce_mean(tf.cast(res, tf.float32))


def mda2(true, pred):
    res = np.equal(np.sign(true[1:] - true[:-1]), np.sign(pred[1:] - pred[:-1]))
    return res


def diff_mape(true, pred):
    return tf.reduce_mean(tf.abs((true - pred) / (true + 0.000001)))


def special_loss(true, pred):
    return tf.abs(((true - pred) ** 2 / pred) / pred) / tf.abs(true)


def visualize(gen, model_1, model_2, p):
    sp = 2
    fig = plt.figure()
    ax = fig.add_subplot(2, 1, 1)
    # ax2 = fig.add_subplot(2, 1, 2, projection='3d')
    # ax3 = fig3.add_subplot(1, 1, 1)
    maes_1 = []
    maes_2 = []
    testX, testY = gen.get_test_data()
    rX, rY = gen.get_raw_data()
    past_window = p + 4

    def animate(i):
        sampleX = testX[i + sp: i + sp + 1]
        sampleY = testY[i + sp:i + sp + 1]

        pred_1 = model_1.predict(sampleX, p)
        pred_2 = model_2.predict(sampleX, p)
        rawX = rX[i + sp:i + sp + 1]
        rawY = rY[i + sp:i + sp + 1]

        sampleX = np.squeeze(sampleX)
        sampleY = np.squeeze(sampleY)
        pred_1 = np.squeeze(pred_1)
        pred_2 = np.squeeze(pred_2)
        rawX = np.squeeze(rawX)
        rawY = np.squeeze(rawY)

        sampleX = np.reshape(sampleX, (-1, gen.input_dim[1] * gen.input_dim[2]))
        sampleY = np.reshape(sampleY, (-1, gen.input_dim[1] * gen.input_dim[2]))
        pred_1 = np.reshape(pred_1, (-1, gen.input_dim[1] * gen.input_dim[2]))
        pred_2 = np.reshape(pred_2, (-1, gen.input_dim[1] * gen.input_dim[2]))
        rawX = np.reshape(rawX, (-1, gen.input_dim[1] * gen.input_dim[2]))
        rawY = np.reshape(rawY, (-1, gen.input_dim[1] * gen.input_dim[2]))

        # sampleX = gen.stdizer.inverse_transform(sampleX)
        # sampleY = gen.stdizer.inverse_transform(sampleY)
        # pred_1 = gen.stdizer.inverse_transform(pred_1)
        # pred_2 = gen.stdizer.inverse_transform(pred_2)

        maes_1.append((sampleY[:p] - pred_1) / (sampleY[:p] + 0.00001))
        maes_2.append((sampleY[:p] - pred_2) / (sampleY[:p] + 0.00001))

        C = rawX[-past_window - 1]

        true = np.concatenate([sampleX[-past_window:], sampleY[:p]], axis=0)
        pred_1 = np.concatenate([sampleX[-past_window:], pred_1], axis=0)
        pred_2 = np.concatenate([sampleX[-past_window:], pred_2], axis=0)
        raw = np.concatenate([rawX[-past_window:], rawY[:p]], axis=0)

        # maes_1.append(mda2(true, pred_1))
        # maes_2.append(mda2(true, pred_2))

        # true[0] += C
        # pred_1[0] += C
        # pred_2[0] += C
        #
        # true = np.cumsum(true, axis=0)
        # pred_1 = np.cumsum(pred_1, axis=0)
        # pred_2 = np.cumsum(pred_2, axis=0)

        true = np.reshape(true, (-1, gen.input_dim[1], gen.input_dim[2]))
        pred_1 = np.reshape(pred_1, (-1, gen.input_dim[1], gen.input_dim[2]))
        pred_2 = np.reshape(pred_2, (-1, gen.input_dim[1], gen.input_dim[2]))
        raw = np.reshape(raw, (-1, gen.input_dim[1], gen.input_dim[2]))

        # mae_1 = np.stack(maes_1, axis=0)
        # mae_2 = np.stack(maes_2, axis=0)
        #
        # mae_1 = np.transpose(mae_1, (0, 2, 1))
        # mae_2 = np.transpose(mae_2, (0, 2, 1))
        #
        # mae_1 = np.reshape(mae_1, (mae_1.shape[0] * mae_1.shape[1], mae_1.shape[2]))
        # mae_2 = np.reshape(mae_2, (mae_2.shape[0] * mae_2.shape[1], mae_2.shape[2]))

        # mae_mean_1 = np.mean(mae_mean_1, axis=(1, 2))
        # mae_mean_2 = np.mean(mae_mean_2, axis=(1, 2))

        # mae_stdev_1 = np.sqrt(np.mean((maes_1 - mae_mean_1) ** 2, axis=0))
        # mae_stdev_2 = np.sqrt(np.mean((maes_2 - mae_mean_2) ** 2, axis=0))

        ax.clear()
        # ax2.clear()
        # ax3.clear()

        ax.plot(true[:, 0, 0])
        # ax.plot(raw[:, 1, 0])
        ax.plot(pred_1[:, 0, 0])
        ax.plot(pred_2[:, 0, 0])
        # ax2.set_ylim(-0, 5)
        # ax2.hist(mae_1[:, 0])

        # for c, z, mae in zip(['r', 'g'], [0, 10], [mae_1, mae_2]):
        #     hist, bins = np.histogram(mae, bins=60, range=(-3, 3))
        #     xs = (bins[:-1] + bins[1:]) / 2
        #     ax2.bar(xs, hist, zs=z, zdir='y', color=c, ec=c, alpha=0.3, width=0.1)

        # ax3.plot(mae_stdev_1[:, 0])
        # ax3.plot(mae_stdev_2[:, 0])

    anim = animation.FuncAnimation(fig, animate, frames=3000, interval=1000)
    # anim2 = animation.FuncAnimation(fig2, animate, frames=3000, interval=600)
    # anim3 = animation.FuncAnimation(fig3, animate, frames=3000, interval=600)

    plt.show()


########################CommonParams##############################

train_ratio = 0.8
symbol = [
    "EURUSD"]  # , "EURGBP", "GBPUSD"]#, "USDCHF", "USDJPY", "USDCAD"]#, "NZDUSD", "CHFJPY", "EURAUD", "AUDJPY", "GBPCAD", "GBPJPY", "EURCAD"]
# symbol = ["EURUSD", "EURGBP", "GBPUSD"]
ohlc = ["close", "high", "low"]
ema = None
oscillator = None
ema_list = None
diff = True
predict_count = 4
date = datetime(2019, 8, 9)
n_sample = 92160
timeframe = MetaTrader5.MT5_TIMEFRAME_M15

##################################################################

#######################LSTMHyperParams############################

seq_length_L = 128
output_count_L = 1

LSTM_layers = [50, 50]
n_pp = 100

batch_size_L = 128
epochs_L = 30
#
# gen_L = VaeGen(train_ratio, seq_length_L, output_count_L, symbol, ohlc, date, n_ema=ema, osc_list=oscillator,
#                test_output_count=predict_count,
#                window_step=1,
#                num_samples=n_sample,
#                diff=diff,
#                timeframe=timeframe)

# mfile_L = './SavedModel/RNN/NaiveResidualLSTM.h5'
mfile_L = './SavedModel/WaveNet/WaveNet_datasettest.h5'
model_saver_L = ModelCheckpoint(mfile_L, save_best_only=True, save_weights_only=True)

##################################################################

#######################WaveNetHyperParams#########################

seq_length_W = 32
output_count_W = 16
batch_size_W = 64
epochs_W = 240
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

model_saver_W = ModelCheckpoint(mfile_W, save_best_only=True, save_weights_only=True)

##################################################################

# gen_list = VaeGen(train_ratio, seq_length_W, output_count_W, symbol, ohlc, date, n_ema=ema,
#                   osc_list=oscillator,
#                   test_output_count=predict_count,
#                   window_step=output_count_W,
#                   num_samples=n_sample,
#                   diff=diff,
#                   timeframe=timeframe)
#
# validX, validY = gen_list.get_val_data()
# validX, validY = validX[:, :, 0:1, :], validY[:, :, 0:1, :]

# trainX, trainY = gen_list.get_train_data()
# trainX, trainY = np.transpose(trainX, (0, 2, 1, 3)), np.transpose(trainY, (0, 2, 1, 3))
# trainX_exp = np.reshape(trainX, (trainX.shape[0] * trainX.shape[1], trainX.shape[2], 1, trainX.shape[3],))
# trainY_exp = np.reshape(trainY, (trainY.shape[0] * trainY.shape[1], trainY.shape[2], 1, trainY.shape[3],))
# print(trainX_exp.shape, trainY_exp.shape)

gen_test = VaeGen(train_ratio, seq_length_W, output_count_W, symbol, ohlc, date, n_ema=ema, osc_list=oscillator,
                  ema_list=ema_list,
                  test_output_count=predict_count + 16,
                  window_step=output_count_W,
                  num_samples=n_sample,
                  diff=diff,
                  logtransform=False,
                  timeframe=timeframe)
print(gen_test.input_dim)

validX, validY = gen_test.get_val_data()
trainX_control, trainY_control = gen_test.get_train_data()

##########################init & training#########################

# wavenet1 = WaveNet(n_filter, n_residual, n_skip, n_layer, n_repeat, filter_width=kernel_size, conditional=False,
#                    l2_lambda=l2_lambda_W)
wavenet2 = WaveNet(n_filter, n_residual, n_skip, n_layer, n_repeat, filter_width=kernel_size, conditional=False,
                   l2_lambda=l2_lambda_W)

# wavenet1.compile(gen_test.input_dim, gen_test.output_dim, optimizer=Adam(decay=0.15), loss=huber_loss)
wavenet2.compile(gen_test.input_dim, gen_test.output_dim, optimizer=Adam(decay=0.15), loss=mean_absolute_error,
                 metrics=[diff_mda])
#
# wavenet1.model.fit([trainX_control, trainY_control], trainY_control,
#                    validation_data=([validX, validY], validY),
#                    batch_size=batch_size_W, epochs=epochs_W, callbacks=[model_saver_L])
# wavenet2.model.fit([trainX_control, trainY_control], trainY_control,
#                    validation_data=([validX, validY], validY),
#                    batch_size=batch_size_W, epochs=epochs_W, callbacks=[model_saver_W])

# wavenet1.model.load_weights(mfile_L)
wavenet2.model.load_weights(mfile_W)

##################################################################
testX, testY = gen_test.get_test_data()
print(testX.shape)
print(testY.shape)
pred = wavenet2.predict(testX, 1)
pred = np.squeeze(pred)
Y = np.squeeze(testY[:, 0:1])

print(pred.shape)
print(Y.shape)
pred = gen_test.stdizer.inverse_transform(pred)
Y = gen_test.stdizer.inverse_transform(Y)


def mymda(true, pred):
    """ Mean Directional Accuracy """
    return np.mean(np.sign(true) == np.sign(pred), axis=0)


def mymetrics(Y, pred):
    Y_class = np.sign(Y)
    pred_class = np.sign(pred)
    ones = np.ones_like(Y_class)
    tps = np.logical_and((Y_class == ones), (pred_class == ones))
    tns = np.logical_and((Y_class == -ones), (pred_class == -ones))
    fps = np.logical_and((Y_class == -ones), (pred_class == ones))
    fns = np.logical_and((Y_class == ones), (pred_class == -ones))
    tp = np.sum(np.logical_and((Y_class == ones), (pred_class == ones)), axis=0)
    tn = np.sum(np.logical_and((Y_class == -ones), (pred_class == -ones)), axis=0)
    fp = np.sum(np.logical_and((Y_class == -ones), (pred_class == ones)), axis=0)
    fn = np.sum(np.logical_and((Y_class == ones), (pred_class == -ones)), axis=0)
    accuracy = (tp + tn) / (tp + fp + fn + tn)
    p_precision = tp / (tp + fp)
    n_precision = tn / (tn + fn)
    recall = tp / (tp + fn)
    F1 = 2 * (recall * p_precision) / (recall + p_precision)

    count_uu = 0
    count_ud = 0
    count_du = 0
    count_dd = 0
    total = 0
    for i in range(len(tps)):  # for all timesteps
        if pred_class[i, 1] == 1 and pred_class[i, 2] == 1:  # when both H and L are predicted to go up
            if Y_class[i, 1] == 1 and Y_class[i, 2] == 1:  # both goes up as predicted; the best
                count_uu += 1 # 33%
            if Y_class[i, 1] == 1 and Y_class[i, 2] == -1:  # high up, low down, second best
                count_ud += 1 # 30%
            if Y_class[i, 1] == -1 and Y_class[i, 2] == 1:  # high down, low up, third best
                count_du += 1 # 30%
            if Y_class[i, 1] == -1 and Y_class[i, 2] == -1: #both down, worst
                count_dd += 1
            total += 1

    count_th = 0
    total_th = 0
    for i in range(len(tps)):  # for all timesteps
        if pred[i, 1] < -0.0003 and pred[i, 2] > 0.0003:
            if Y_class[i, 1] == -1 and Y_class[i, 2] == 1: # checking accuracy when the pred value is higher by certain amount
                count_th += 1
            total_th += 1



    print("uu", count_uu/total, "ud", count_ud/total, "du", count_du/total, "dd", count_dd/total, "sumall",
           count_uu/total + count_ud/total + count_du/total + count_dd/total)
    print("threshold acc 0.0001", count_th/total_th)
    print("acc", accuracy, "p_precision", p_precision,
          "recall", recall, "F1", F1)


# print(mymda(validY[:, 0:1], pred))
print(mymetrics(Y, pred))
plt.plot(Y[:, 0])
plt.plot(pred[:, 0])
plt.show()
plt.plot(Y[:, 1])
plt.plot(pred[:, 1])
plt.show()
plt.plot(Y[:, 2])
plt.plot(pred[:, 2])
plt.show()

# visualize(gen_test, wavenet2, wavenet2, predict_count)
