from tensorflow.python.keras.layers import *
import tensorflow.python.keras as keras
from tensorflow.python.keras.callbacks import *
from tensorflow.python.keras.regularizers import *
from datetime import datetime
import MetaTrader5
import matplotlib.pyplot as plt
from tensorflow.python.keras.optimizers import Adam

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'


class WaveVAE:
    def __init__(self, latent_dim, l2_lambda):
        encoder = keras.models.Sequential()
        encoder.add(keras.layers.Conv2D(filters=8, kernel_size=(2, 1), strides=(2, 1), activation='selu',
                                        kernel_regularizer=l2(l2_lambda)))
        encoder.add(keras.layers.Conv2D(filters=8, kernel_size=(2, 1), strides=(2, 1), activation='selu',
                                        kernel_regularizer=l2(l2_lambda)))  # 250, 3, 8
        encoder.add(keras.layers.Conv2D(filters=16, kernel_size=(2, 1), strides=(2, 1), activation='selu',
                                        kernel_regularizer=l2(l2_lambda)))  # 125, 3, 8
        encoder.add(keras.layers.Conv2D(filters=32, kernel_size=(2, 1), strides=(2, 1), activation='selu',
                                        kernel_regularizer=l2(l2_lambda)))
        encoder.add(keras.layers.Conv2D(filters=32, kernel_size=(2, 1), strides=(2, 1), activation='selu',
                                        kernel_regularizer=l2(l2_lambda)))
        encoder.add(keras.layers.Conv2D(filters=64, kernel_size=(2, 1), strides=(2, 1), activation='selu',
                                        kernel_regularizer=l2(l2_lambda)))
        encoder.add(keras.layers.Conv2D(filters=64, kernel_size=(2, 1), strides=(2, 1), activation='selu',
                                        kernel_regularizer=l2(l2_lambda)))
        encoder.add(keras.layers.Conv2D(filters=128, kernel_size=(2, 1), strides=(2, 1), activation='selu',
                                        kernel_regularizer=l2(l2_lambda)))
        encoder.add(keras.layers.Conv2D(filters=128, kernel_size=(2, 1), strides=(2, 1), activation='selu',
                                        kernel_regularizer=l2(l2_lambda)))
        encoder.add(keras.layers.Conv2D(filters=256, kernel_size=(2, 1), strides=(2, 1), activation='selu',
                                        kernel_regularizer=l2(l2_lambda)))
        encoder.add(keras.layers.Conv2D(filters=256, kernel_size=(2, 1), strides=(2, 1), activation='selu',
                                        kernel_regularizer=l2(l2_lambda)))
        encoder.add(keras.layers.Conv2D(filters=512, kernel_size=(1, 3), strides=(1, 1), activation='selu',
                                        kernel_regularizer=l2(l2_lambda)))
        encoder.add(Flatten())
        encoder.add(Dense(300, activation="relu"))

        decoder = keras.models.Sequential()
        decoder.add(keras.layers.Conv2DTranspose(filters=512, kernel_size=(1, 3), strides=(1, 1), activation='selu',
                                                 kernel_regularizer=l2(l2_lambda)))
        decoder.add(keras.layers.Conv2DTranspose(filters=256, kernel_size=(2, 1), strides=(2, 1), activation='selu',
                                                 kernel_regularizer=l2(l2_lambda)))
        decoder.add(keras.layers.Conv2DTranspose(filters=128, kernel_size=(2, 1), strides=(2, 1), activation='selu',
                                                 kernel_regularizer=l2(l2_lambda)))
        decoder.add(keras.layers.Conv2DTranspose(filters=128, kernel_size=(2, 1), strides=(2, 1), activation='selu',
                                                 kernel_regularizer=l2(l2_lambda)))
        decoder.add(keras.layers.Conv2DTranspose(filters=64, kernel_size=(2, 1), strides=(2, 1), activation='selu',
                                                 kernel_regularizer=l2(l2_lambda)))
        decoder.add(keras.layers.Conv2DTranspose(filters=64, kernel_size=(2, 1), strides=(2, 1), activation='selu',
                                                 kernel_regularizer=l2(l2_lambda)))
        decoder.add(keras.layers.Conv2DTranspose(filters=32, kernel_size=(2, 1), strides=(2, 1), activation='selu',
                                                 kernel_regularizer=l2(l2_lambda)))
        decoder.add(keras.layers.Conv2DTranspose(filters=32, kernel_size=(2, 1), strides=(2, 1), activation='selu',
                                                 kernel_regularizer=l2(l2_lambda)))
        decoder.add(keras.layers.Conv2DTranspose(filters=16, kernel_size=(2, 1), strides=(2, 1), activation='selu',
                                                 kernel_regularizer=l2(l2_lambda)))
        decoder.add(keras.layers.Conv2DTranspose(filters=8, kernel_size=(2, 1), strides=(2, 1), activation='selu',
                                                 kernel_regularizer=l2(l2_lambda)))
        decoder.add(keras.layers.Conv2DTranspose(filters=4, kernel_size=(2, 1), strides=(2, 1), activation='selu',
                                                 kernel_regularizer=l2(l2_lambda)))
        decoder.add(keras.layers.Conv2DTranspose(filters=2, kernel_size=(2, 1), strides=(2, 1), activation='selu',
                                                 kernel_regularizer=l2(l2_lambda)))
        decoder.add(keras.layers.Conv2DTranspose(filters=1, kernel_size=(1, 1)))

        self.latent_dim = latent_dim
        self.encoder = encoder
        self.decoder = decoder

    def compile(self, input_dim, optimizer):
        x = Input(shape=(input_dim[0], input_dim[1], input_dim[2]))
        enc_out = self.encoder(x)

        z_m = keras.layers.Dense(self.latent_dim, name="mean")(enc_out)
        z_logvar = keras.layers.Dense(self.latent_dim, name="logvar")(enc_out)

        def sampling(z_mean, z_log_sigma):
            epsilon = K.random_normal(shape=(K.shape(z_mean)[0], self.latent_dim),
                                      mean=0., stddev=0.1)
            return z_mean + z_log_sigma * epsilon

        z = Lambda(lambda x: sampling(x[0], x[1]))([z_m, z_logvar])
        z_input = keras.Input(shape=(self.latent_dim,))

        expand = Lambda(lambda l: K.expand_dims(K.expand_dims(l, axis=1), axis=1))

        z_expand = expand(z)
        z_input_expand = expand(z_input)

        out = self.decoder(z_expand)
        dec_out = self.decoder(z_input_expand)

        def loss(true, pred):
            kl_loss = - 0.5 * K.mean(
                1 + z_logvar - K.square(z_m) - K.exp(z_logvar))

            reconstruction_loss = K.mean(keras.losses.mean_absolute_error(true, pred))
            model_loss = kl_loss + reconstruction_loss

            return model_loss

        self.VAE = keras.models.Model(x, out)
        self.VAE_encoder = keras.models.Model(x, z)
        self.VAE_decoder = keras.models.Model(z_input, dec_out)
        self.VAE.compile(optimizer=optimizer, loss=loss)
        self.VAE.summary()


if __name__ == "__main__":
    symbol = ["EURUSD", "EURGBP", "GBPUSD"]
    batch_size = 128
    epochs = 64
    latent_dim = 200

    mfile = './SavedModel/VAE/WaveVAE.h5'
    model_saver = ModelCheckpoint(mfile, save_best_only=True, save_weights_only=True)
    # tensorboard --host 127.0.0.1 --logdir=D:\Projects\AI\Summary\Wavenet\

    from DataCookers.VAEdataset import VaeGen

    gen = VaeGen(0.8, 2048, 1, symbol, datetime(2019, 5, 26), num_samples=92160,
                 timeframe=MetaTrader5.MT5_TIMEFRAME_M15)

    dataset = np.stack([gen.dataset[2048 * i:2048 * (i + 1)] for i in range(int(len(gen.dataset) / 2048))])
    print(dataset.shape)

    VAE = WaveVAE(latent_dim, 0.0001)
    VAE.compile(gen.input_dim, Adam(clipvalue=1))
    # VAE.VAE.fit(gen.trainX, gen.trainX, batch_size=batch_size, epochs=epochs, validation_data=(gen.validX, gen.validX),
    #             callbacks=[model_saver], verbose=1)
    VAE.VAE.load_weights(mfile)

    for elem in dataset:
        e = np.expand_dims(elem, axis=0)
        pred = VAE.VAE.predict(e)
        plt.plot(np.squeeze(e))
        plt.plot(np.squeeze(pred))
        plt.show()




    # for i in range(10):
    #     rand = int(random.uniform(0, 7000))
    #     # rand2 = random.uniform(0, 2)
    #     true = gen.validX[rand:rand + 1]
    #     raw = gen.rawX[rand:rand + 1]
    #     recon = gen.stdizer.inverse_transform(np.squeeze(VAE.VAE.predict(true)))
    #     plt.plot(np.squeeze(raw))
    #     plt.plot(recon)
    #     plt.show()
