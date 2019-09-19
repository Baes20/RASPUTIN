from __future__ import absolute_import, division, print_function

import tensorflow as tf
from tensorflow.python.keras.layers import *
from tensorflow.python.keras.models import *
from tensorflow.python.keras.optimizers import *
from tensorflow.python.keras.callbacks import *
import matplotlib.pyplot as plt
from datetime import datetime
import MetaTrader5
from matplotlib import animation
from DataCookers.VAEdataset import VaeGen


# os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'


class SeqVAE:
    def __init__(self, encoder_layers, decoder_layers, latent_dim, epsilon_std=0.1):
        self.encoder = CuDNNLSTM(encoder_layers[0])
        self.decoder = CuDNNLSTM(decoder_layers[0], return_sequences=True)
        self.encoder_mean = Dense(latent_dim)
        self.encoder_logvar = Dense(latent_dim)
        self.latent_dim = latent_dim
        self.e_std = epsilon_std
        self.lat_dim = latent_dim
        self.epsilon_std = epsilon_std

    def compile(self, input_dim, output_dim, optimizer):
        input = Input(shape=(None, input_dim[1], input_dim[2]))
        input_flat = Reshape(target_shape=(-1, input_dim[1] * input_dim[2]))(input)

        projection = Dense(output_dim[1] * output_dim[2])

        x = self.encoder(input_flat)
        self.z_mean = self.encoder_mean(x)
        self.z_logstd = self.encoder_logvar(x)

        latent_dim = self.latent_dim
        epsilon_std = self.epsilon_std

        @tf.function
        def sampling(z_mean, z_log_sigma):
            epsilon = K.random_normal(shape=(K.shape(z_mean)[0], latent_dim),
                                      mean=0., stddev=epsilon_std)
            return z_mean + z_log_sigma * epsilon

        z = Lambda(lambda x: sampling(x[0], x[1]))([self.z_mean, self.z_logstd])
        z_input = Input(shape=(self.latent_dim,))
        z_decoder = RepeatVector(output_dim[0])(z)
        _z_decoder = RepeatVector(output_dim[0])(z_input)

        decoder_out = self.decoder(z_decoder)
        _decoder_out = self.decoder(_z_decoder)

        decoder_out = projection(decoder_out)
        _decoder_out = projection(_decoder_out)

        x_prime = Reshape(target_shape=(-1, input_dim[1], input_dim[2]))(decoder_out)
        _x_prime = Reshape(target_shape=(-1, input_dim[1], input_dim[2]))(_decoder_out)
        x_prime = Lambda(lambda x: x[:, :output_dim[0], :], name="train_prediction")(x_prime)

        @tf.function
        def loss(true, pred):
            kl_loss = - 0.5 * K.mean(
                1 + self.z_logstd - tf.square(self.z_mean) - K.exp(self.z_logstd))

            reconstruction_loss = K.mean(tf.keras.losses.mean_absolute_error(true, pred))
            model_loss = kl_loss + reconstruction_loss

            return model_loss

        self.model = Model(input, x_prime)
        self.encoder_model = Model(input, z)
        self.decoder_model = Model(z_input, _x_prime)

        self.model.compile(optimizer=optimizer, loss=loss)
        self.model.summary()

    def predict(self, input):
        return self.model.predict(input)

    def encode(self, input):
        return self.encoder_model.predict(input)

    def decode(self, z):
        return self.decoder_model.predict(z)


if __name__ == "__main__":
    recon_length = 64
    pred_length = 4
    input_dim = [recon_length, 3, 3]
    output_dim_for_train = [recon_length, 3, 3]
    output_dim_for_test = [recon_length + pred_length, 3, 3]
    encoder_layers = [100, 100]
    decoder_layers = [100, 100]
    batch_size = 128
    symbol_list = ["EURUSD", "EURGBP", "GBPUSD"]
    epoch = 300

    latent_dim = 100

    gen = VaeGen(0.8, recon_length, pred_length, symbol_list=symbol_list,
                                  last_date=datetime(2019, 5, 28),
                                  num_samples=5000, timeframe=MetaTrader5.MT5_TIMEFRAME_D1)

    trainX = gen.trainX
    validX = gen.validX
    validY = gen.validY

    mfile = '.\SavedModel\VAE/VAE.h5'
    summarydir = ".\Summary\VAE"

    model_saver = ModelCheckpoint(mfile, save_best_only=True, save_weights_only=True)
    tensorboard = TensorBoard(log_dir=summarydir, write_graph=True)

    vae = SeqVAE(encoder_layers, decoder_layers, latent_dim)
    vae.compile(input_dim, output_dim=output_dim_for_train, optimizer=Adam(lr=0.001, epsilon=1e-9))

    # vae.model.fit(trainX, trainX, batch_size=batch_size, epochs=epoch,
    #               validation_data=(gen.validX, gen.validX), callbacks=[model_saver])

    vae.model.load_weights(mfile)


    def visualize(gen, model):
        sp = 600
        fig = plt.figure()
        ax = fig.add_subplot(1, 1, 1)

        def animate(i):
            if i % 10 == 0:
                sampleX = gen.validX[i + sp: i + 1 + sp]
                sampleY = gen.validY[i + sp: i + 1 + sp]
                pred = model.predict(sampleX)
                ax.clear()
                ax.plot(np.concatenate([np.squeeze(sampleX), np.squeeze(sampleY)], axis=0)[:, 0])
                ax.plot(np.squeeze(pred)[:, 0])
                ax.axvline(x=recon_length-1)

        anim = animation.FuncAnimation(fig, animate, frames=1000, interval=50)
        plt.show()


    visualize(gen, vae)

# EXTREMELY SUCCESSFUL!!
