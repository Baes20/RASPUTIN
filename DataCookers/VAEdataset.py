from __future__ import absolute_import, division, print_function

import os

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

# TensorFlow and tf.keras
import matplotlib.pyplot as plt
# Helper libraries
import numpy as np
from DataCookers.MT5DataGetter import MT5DataGetter
from sklearn import preprocessing
import MetaTrader5
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from statsmodels.tsa.stattools import adfuller
import pandas as pd
from DataCookers.fracdiff import fast_fracdiff, frac_diff_ffd
from seaborn import distplot
from matplotlib import animation
from scipy.stats import gaussian_kde
import random


def adf_test(timeseries):
    # Perform Dickey-Fuller test:
    print('Results of Dickey-Fuller Test:')
    dftest = adfuller(timeseries, autolag='AIC')
    dfoutput = pd.Series(dftest[0:4], index=['Test Statistic', 'p-value', '#Lags Used', 'Number of Observations Used'])
    for key, value in dftest[4].items():
        dfoutput['Critical Value (%s)' % key] = value
    print(dfoutput)


class VaeGen(object):
    @staticmethod
    def exp_moving_avg(data, timestep):
        ema = []
        k = 2 / (timestep + 1)
        ema.append(data[0])
        for symbols in data[1:]:
            res = symbols * k + ema[-1] * (1 - k)
            ema.append(res)
        return np.stack(ema)

    @staticmethod
    def createTestData_nparray(data, seqLength, predLength=1):
        i = 0
        dataX = []
        dataY = []
        while (i < (len(data) - seqLength - predLength)):
            dataX.append(data[i:i + seqLength])
            dataY.append(data[i + seqLength:(i + seqLength + predLength)])
            i += predLength

        return np.array(dataX), np.array(dataY)

    @staticmethod
    def createTrainData_nparray(data, seqLength, predLength=1, stride=1):
        i = 0
        dataX = []
        dataY = []
        while (i < (len(data) - seqLength - predLength)):
            dataX.append(data[i:i + seqLength])
            dataY.append(data[i + seqLength:(i + seqLength + predLength)])
            i += stride

        return np.array(dataX), np.array(dataY)

    @staticmethod
    def split_to_blocks(data, seqLength, predLength=1, stride=1):
        i = 0
        dataX = []
        while (i < (len(data) - seqLength - predLength)):
            dataX.append(data[i:i + seqLength + predLength])
            i += stride

        return np.array(dataX)

    def __init__(self, train_ratio,
                 seq_length,
                 output_count,
                 symbol_list,
                 ohlc_list,
                 last_date,
                 n_ema=None,
                 ema_list=None,
                 osc_list=None,
                 test_output_count=16,
                 num_samples=99999,
                 window_step=1,
                 mostrecent=False,
                 diff=True,
                 logtransform=False,
                 preprocess=True,
                 timeframe=MetaTrader5.MT5_TIMEFRAME_M1):

        self.seq_length = seq_length
        self.output_count = output_count
        self.window_step = window_step
        self.test_output_count = test_output_count

        if mostrecent:
            raw_datasets = MT5DataGetter(symbol_list).getmostrecent(num_samples, timeframe)
        else:
            raw_datasets = MT5DataGetter(symbol_list).getcandledata(last_date, num_samples, timeframe)

        dataset = []
        d_raw = []
        self.get_index_from_dict = {}
        index = 0
        for symbol, symbol_str in zip(raw_datasets, symbol_list):
            flist = []
            for ohlc in ohlc_list:
                if ema_list is not None:
                    for ema_num in ema_list:
                        f = np.expand_dims(np.expand_dims(symbol[ohlc].values, axis=-1), axis=-1)
                        d_raw.append(f)
                        f = self.exp_moving_avg(f, ema_num)
                        if diff:
                            f = np.diff(f, axis=0, append=f[-1:])
                            # f = np.expand_dims(np.expand_dims(fast_fracdiff(np.squeeze(f), d=0.4), axis=-1),axis=-1)
                        flist.append(f)
                        self.get_index_from_dict[str(symbol_str) + "_" + str(ohlc) + "_ema" + str(ema_num)] = index
                        index += 1
                elif n_ema is not None:
                    f = np.expand_dims(np.expand_dims(symbol[ohlc].values, axis=-1), axis=-1)
                    d_raw.append(f)
                    f = self.exp_moving_avg(f, n_ema)
                    if diff:
                        f = np.diff(f, axis=0, append=f[-1:])
                        # f = np.expand_dims(np.expand_dims(fast_fracdiff(np.squeeze(f), d=0.4), axis=-1),axis=-1)
                    flist.append(f)
                    self.get_index_from_dict[str(symbol_str) + "_" + str(ohlc)] = index
                    index += 1
                else:
                    f = np.expand_dims(np.expand_dims(symbol[ohlc].values, axis=-1), axis=-1)
                    d_raw.append(f)
                    if diff:
                        f = np.diff(f, axis=0, append=f[-1:])
                        # np.expand_dims(np.expand_dims(fast_fracdiff(np.squeeze(f), d=0.4), axis=-1),axis=-1)
                    flist.append(f)
                    self.get_index_from_dict[str(symbol_str) + "_" + str(ohlc)] = index
                    index += 1

            if osc_list is not None:
                for oscillator in osc_list:
                    if oscillator == "atr":
                        true_range = symbol["high"].values - symbol["low"].values
                        avg_true_range = self.exp_moving_avg(true_range, 14)
                        f = np.expand_dims(np.expand_dims(avg_true_range, axis=-1), axis=-1)
                        flist.append(f)
                        d_raw.append(f)

                    if oscillator == "macd":
                        short_ema = self.exp_moving_avg(symbol["close"].values, 20)
                        long_ema = self.exp_moving_avg(symbol["close"].values, 60)
                        macd = short_ema - long_ema
                        macd_signal = self.exp_moving_avg(macd, 5)
                        macd_oscillator = macd - macd_signal
                        f = np.expand_dims(np.expand_dims(macd_oscillator, axis=-1), axis=-1)
                        flist.append(f)
                        d_raw.append(f)

                    self.get_index_from_dict[symbol_str + "_" + oscillator] = index
                    index += 1

            hlc = np.concatenate(flist, axis=-1)
            dataset.append(hlc)

        print(self.get_index_from_dict)

        dataset = np.concatenate(dataset, axis=1)  # (timesteps, markets, features)
        d_raw = np.concatenate(d_raw, axis=1)
        shape = dataset.shape
        self.input_dim = [seq_length, shape[1], shape[2]]
        self.output_dim = [output_count, shape[1], shape[2]]

        if logtransform:
            dataset = np.log(dataset)

        dataset = np.reshape(dataset, [dataset.shape[0], dataset.shape[1] * dataset.shape[2]])
        prev_dataset = dataset

        # self.output_stdizer = preprocessing.RobustScaler().fit(prev_dataset[:, 0:1])
        self.stdizer = preprocessing.RobustScaler()
        self.stdizer.fit(prev_dataset)

        if preprocess:
            dataset = self.stdizer.transform(dataset)

        self.dataset = dataset
        self.train_size = int(len(self.dataset) * train_ratio)
        plt.plot(dataset)
        plt.show()
        self.dataset = np.reshape(dataset, [shape[0], shape[1], shape[2]])
        self.d_raw = d_raw[self.train_size:]

    def get_train_data(self):
        train = self.split_to_blocks(self.dataset[:self.train_size],
                                     self.seq_length,
                                     self.output_count,
                                     stride=self.window_step)

        return train

    def get_val_data(self):
        valid = self.split_to_blocks(self.dataset[self.train_size:],
                                     self.seq_length,
                                     self.output_count,
                                     stride=self.window_step)

        return valid

    def get_test_data(self):
        valid = self.split_to_blocks(self.dataset[self.train_size:],
                                     self.seq_length,
                                     self.output_count,
                                     stride=1)
        X, Y = valid[:, :self.seq_length], valid[:, self.seq_length:]
        return X, Y

    def get_X_and_Y(self, data):
        return data[:, :self.seq_length], data[:, self.seq_length:]


