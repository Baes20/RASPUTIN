from AI.FXTM_Predictor_Wavenet.WaveNet import WaveNet
from tensorflow.python.keras.callbacks import *
from tensorflow.python.keras.optimizers import Adam
from DataCookers.FXTMdataset import MarketDataGenerator
import os
from datetime import datetime
import MetaTrader5
import matplotlib.pyplot as plt
from matplotlib import animation

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'


def mda(actual: np.ndarray, predicted: np.ndarray):
    """ Mean Directional Accuracy """
    return np.mean((np.sign(actual[1:] - actual[:-1]) == np.sign(predicted[1:] - predicted[:-1])).astype(int))


def visualize(gen, model, cond, pred_length=4):
    sp = 0
    fig = plt.figure()
    ax = fig.add_subplot(2, 1, 1)
    ax2 = fig.add_subplot(2, 1, 2)

    def animate(i):
        if i%15 == 0:
            sampleX = gen.testX[i + sp: i + 1 + sp]
            pred = model.predict(sampleX, pred_length)

            sampleX = np.squeeze(sampleX)
            pred = np.squeeze(pred)

            sampleY = gen.testY[i + sp:i + sp + 1]
            sampleY = np.squeeze(sampleY)

            # samplecond = cond[i + sp: i + sp + 1]
            rawX = gen.rawX[i + sp:i + sp + 1]
            rawX = np.squeeze(rawX)

            rawY = gen.rawY[i + sp:i + sp + 1]
            rawY = np.squeeze(rawY)

            vaeX = gen.vaeX[i + sp:i + sp + 1]
            vaeX = np.squeeze(vaeX)
            C = vaeX[-pred_length - 1]

            # C = rawX[-pred_length - 1]

            sampleX = gen.stdizer.inverse_transform(sampleX)
            sampleY = gen.stdizer.inverse_transform(sampleY)
            pred = gen.stdizer.inverse_transform(pred)

            true = np.concatenate([sampleX[-pred_length:], sampleY[:pred_length]], axis=0)
            pred = np.concatenate([sampleX[-pred_length:], pred[:]], axis=0)

            true[0] += C
            pred[0] += C

            true = np.cumsum(true, axis=0)
            pred = np.cumsum(pred, axis=0)
            raw = np.concatenate([rawX[-pred_length:], rawY[:pred_length]], axis=0)

            ax.clear()
            ax2.clear()

            ax2.plot(np.diff(true, axis=0)[:, 0])
            ax2.plot(np.diff(pred, axis=0)[:, 0])

            ax.axvline(x=predict_count - 1)
            ax.plot(true[:, 0])
            ax.plot(pred[:, 0])
            ax.plot(raw[:, 0])

    anim = animation.FuncAnimation(fig, animate, frames=3000, interval=50)
    plt.show()


#######################HyperParams#########################
train_ratio = 0.8
seq_length = 512
output_count = 1024
predict_count = 2
batch_size = 1
epoch = 100
l2_lambda = 0.00001

n_residual = 32
n_filter = 32
n_skip = 128
n_layer = 9
n_repeat = 1
kernel_size = 3
symbol = ["EURUSD"]

Hparam = '-' + str(n_filter) + '-' + str(n_residual) + '-' + str(n_skip) + '-' + str(n_layer) + '-' + str(n_repeat)

# p = {
#     'output_count': [16, 64, 128, 256],
#     'batch_size': [16, 32, 64, 128],
#     'dropout_rate': [0.1, 0.2, 0.3],
#     'n_pp': [4, 8, 16, 32, 64],
#     'n_filter': [4, 8, 16, 32, 64],
#     'n_fc': [16, 32, 64, 128, 256],
#     'n_layer': [8, 9, 10],
#     'n_repeat': [1, 2, 3],
#     'kernel_size': [2, 3, 4],
#     'lr': [0.01, 0.001, 0.0001],
#     'l2_lambda': [0.01, 0.001, 0.0001]
# }
###########################################################

mfile = './SavedModel/WaveNet/WaveNet' + Hparam + '.h5'
best = './SavedModel/WaveNet/Best/WaveNet' + Hparam + '-H4_best.h5'
summarydir = "./Summary/Wavenet/" + Hparam + "/"
tensorboard = TensorBoard(log_dir=summarydir, write_graph=True, histogram_freq=1)
model_saver = ModelCheckpoint(mfile, save_best_only=True, save_weights_only=True)
# tensorboard --host 127.0.0.1 --logdir=D:\Projects\AI\Summary\Wavenet\

gen = MarketDataGenerator(train_ratio, seq_length, output_count, symbol, datetime(2019, 8, 2), num_samples=92160,
                          testing=False,
                          timeframe=MetaTrader5.MT5_TIMEFRAME_M1)
#
# train_cond = pd.read_csv("./Datasets/3FXTM15M_1024_2019_5_10_80000_train.csv", index_col=0).values
# valid_cond = pd.read_csv("./Datasets/3FXTM15M_1024_2019_5_10_80000_valid.csv", index_col=0).values
# test_cond = pd.read_csv("./Datasets/3FXTM15M_1024_2019_5_10_80000_test.csv", index_col=0).values

trainX = gen.trainX
trainY = gen.trainY
validX = gen.validX
validY = gen.validY
testX = gen.testX
testY = gen.testY

wavenet = WaveNet(n_filter, n_residual, n_skip, n_layer, n_repeat, filter_width=kernel_size, conditional=False,
                  l2_lambda=l2_lambda)
wavenet.compile(gen.input_dim, gen.output_dim, optimizer=Adam(), loss='mae', cond_latent_dim=10)
# wavenet.model.fit([trainX, trainY], trainY,
#                   validation_data=([validX, validY], validY),
#                   batch_size=batch_size, epochs=epoch, callbacks=[model_saver])
wavenet.model.load_weights(mfile)

visualize(gen, wavenet, pred_length=predict_count, cond=None)

# def WN(x_train, y_train, x_val, y_val, par):
#     wavenet = WaveNet(par['n_filter'],
#                       par['dropout_rate'],
#                       par['n_pp'],
#                       par['n_fc'],
#                       par['n_layer'],
#                       par['n_repeat'],
#                       filter_width=par['kernel_size'],
#                       l2_lambda=par['l2_lambda'])
#     wavenet.compile(gen.input_dim, gen.output_dim, optimizer=Adam(lr=par['lr']), mode=0, default_loss=huber_loss)
#     out = wavenet.model_train.fit([x_train, y_train], y_train, batch_size=par['batch_size'], epochs=epoch,
#                                   validation_data=([x_val, y_val], y_val), verbose=2)
#
#     return out, wavenet.model_train
#
#
# history = ta.Scan(trainX, trainY, params=p, model=WN, x_val=validX, y_val=validY, val_split=0,
#                   reduction_method='correlation', dataset_name='3_M15', experiment_no='1',
#                   grid_downsample=0.01,)
