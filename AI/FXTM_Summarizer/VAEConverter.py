from __future__ import absolute_import, division, print_function

import os

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
from AI.FXTM_Predictor_RNN.LSTMVAE import SeqVAE
from DataCookers.FXTMdataset import MarketDataGenerator
import tensorflow.python.keras as keras
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime

mfile = '.\SavedModel\VAE/VAE.h5'
mfile_enc = '.\SavedModel\VAE\VAE_encoder.h5'
mfile_dec = '.\SavedModel\VAE\VAE_decoder.h5'
mfile_arch = '.\SavedModel\VAE\VAE_arch.json'
mfile_enc_arch = '.\SavedModel\VAE\VAE_encoder_arch.json'
mfile_dec_arch = '.\SavedModel\VAE\VAE_decoder_arch.json'
summarydir = ".\Summary\VAE"

input_dim = [16, 3, 1]
encoder_layers = [400]
decoder_layers = [400, input_dim[1] * input_dim[2]]
batch_size = 32
epoch = 10
symbol_list = ["EURGBP", "EURUSD", "GBPUSD"]
latent_dim = 30

gen = MarketDataGenerator(0.8, 16, 8, batch_size, symbol_list, datetime(2019, 4, 20), num_samples=90000)
whole = gen.og_dataset
wholeX, _ = gen.createTestData_nparray(whole, 16, 16)
wholeX = np.expand_dims(wholeX,axis=-1)
print(wholeX.shape)

vae = SeqVAE(input_dim, encoder_layers, decoder_layers, latent_dim)
vae.compile(input_dim, keras.optimizers.adam(epsilon=0.0001))

vae.model.load_weights(mfile)
encoded = vae.encode(wholeX)
decoded = vae.decode(encoded)
pd.DataFrame(encoded).to_csv("./Datasets/3FXTM1M_exp_moving10_delta_norm_encoded.csv")
decoded = np.reshape(decoded, newshape=[-1, input_dim[1] * input_dim[2]])
testX = np.reshape(whole, newshape=[-1, input_dim[1] * input_dim[2]])


plt.plot(testX)
plt.plot(decoded)
plt.show()
