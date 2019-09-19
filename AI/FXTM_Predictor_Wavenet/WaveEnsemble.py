from tensorflow.python.keras.callbacks import *
from tensorflow.python.keras.optimizers import Adam
from tensorflow.python.keras.losses import *
import os
from AI.FXTM_Predictor_Wavenet.CondWaveNet import CondWaveNet
import gc

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'


class CondWaveEnsemble:
    def __init__(self, generator, symbols, n_conv_filters, n_fc, num_layer, n_repeat=1, filter_width=2,
                 l2_lambda=0.001, dropout_rate=0.2):
        self.symbols = symbols

        self.ensemble = [CondWaveNet(n_conv_filters, dropout_rate, n_fc, num_layer, n_repeat,
                                     filter_width=filter_width, l2_lambda=l2_lambda) for _ in range(len(symbols))]

        self.gen = generator

        self.Hparam = '-' + str(dropout_rate) + '-' + str(n_conv_filters) + '-' + str(n_fc) + '-' + str(num_layer) \
                      + '-' + str(n_repeat) + '-' + str(filter_width) + '-' + str(l2_lambda)

        self.mfiles = []
        for symbol in symbols:
            savefile = './SavedModel/CondWaveNet/CondWaveNet-' + str(symbol) + self.Hparam + '.h5'
            self.mfiles.append(savefile)

        self.already_trained = False

    def train(self, batch_size, epoch, optim=Adam(0.001), loss=huber_loss):
        trainX = self.gen.trainX
        trainY = self.gen.trainY
        validX = self.gen.validX
        validY = self.gen.validY

        for i in range(len(self.symbols)):
            # summarydir = './Summary/CondWavenet/' + str(self.symbols[i]) + self.Hparam + '/'
            # tensorboard = TensorBoard(log_dir=summarydir, write_graph=True, histogram_freq=1, write_images=True)

            print("Model for " + self.symbols[i] + ", training....")
            model_saver = ModelCheckpoint(self.mfiles[i], save_best_only=True, save_weights_only=True)
            self.ensemble[i].compile(self.gen.input_dim, self.gen.output_dim,
                                     optimizer=optim, default_loss=loss, main_input_index=i)
            self.ensemble[i].model_train.fit([trainX, trainY], trainY[..., i:i + 1, 0:1], batch_size=batch_size,
                                             epochs=epoch, callbacks=[model_saver],
                                             validation_data=([validX, validY], validY[..., i:i + 1, 0:1]), verbose=2)

            K.clear_session()
            gc.collect()

        self.already_trained = True

    def predict(self, input, prediction_length):  # input= (1, timestep, symbols, features)

        if not self.already_trained:

            for i in range(len(self.ensemble)):
                self.ensemble[i].compile(self.gen.input_dim, self.gen.output_dim,
                                         optimizer='adam', main_input_index=i)
                self.ensemble[i].model_train.load_weights(self.mfiles[i])

        inp = np.copy(input)
        pred = np.zeros_like(inp)
        pred = pred[:, :prediction_length]

        for i in range(prediction_length):
            encX = inp[:, :-1]
            decX = np.concatenate([inp[:, -1:], np.zeros_like(inp)[:, 0:1]], axis=1)

            last_pred = []
            for model in self.ensemble:
                last_pred.append(model.model_train.predict([encX, decX])[:, -1, :, :])  # (1(batch), 1(sym), 1(feat))
            last_pred = np.concatenate(last_pred, axis=1)

            pred[:, i] = last_pred

            inp = np.concatenate([inp, pred[:, i:i + 1]], axis=1)

        self.already_trained = True
        return pred
