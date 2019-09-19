from __future__ import absolute_import, division, print_function

import os
from scipy.sparse import csr_matrix
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

# TensorFlow and tf.keras
import matplotlib.pyplot as plt
# Helper libraries
import numpy as np
from DataCookers.MT5DataGetter import MT5DataGetter
from sklearn import preprocessing
import MetaTrader5
from datetime import datetime
from tensorflow.python.keras.utils import to_categorical


class FXTMCategorical(object):
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
        data = csr_matrix.todense(data)
        i = 0
        dataX = []
        dataY = []
        while (i < (len(data) - seqLength - predLength)):
            dataX.append(data[i:i + seqLength])
            dataY.append(data[i + seqLength:(i + seqLength + predLength)])
            i += stride

        return np.array(dataX), np.array(dataY)

    def __init__(self, train_ratio,
                 seq_length,
                 output_count,
                 symbol_list,
                 ohlc_list,
                 last_date,
                 n_ema=5,
                 ema_list=None,
                 osc_list=None,
                 test_output_count=16,
                 num_samples=99999,
                 window_step=1,
                 mostrecent=False,
                 diff=True,
                 num_bins=256,
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
                            f = np.diff(f, axis=0, prepend=f[0:1])
                        flist.append(f)
                        self.get_index_from_dict[str(symbol_str) + "_" + str(ohlc) + "_ema" + str(ema_num)] = index
                        index += 1
                elif n_ema is not None:
                    f = np.expand_dims(np.expand_dims(symbol[ohlc].values, axis=-1), axis=-1)
                    d_raw.append(f)
                    f = self.exp_moving_avg(f, n_ema)
                    if diff:
                        f = np.diff(f, axis=0, prepend=f[0:1])
                    flist.append(f)
                    self.get_index_from_dict[str(symbol_str) + "_" + str(ohlc)] = index
                    index += 1
                else:
                    f = np.expand_dims(np.expand_dims(symbol[ohlc].values, axis=-1), axis=-1)
                    d_raw.append(f)
                    if diff:
                        f = np.diff(f, axis=0, prepend=f[0:1])
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

        self.dataset = np.reshape(dataset, [dataset.shape[0], dataset.shape[1] * dataset.shape[2]])
        # plt.plot(self.dataset)
        # plt.show()
        self.digitizer = preprocessing.KBinsDiscretizer(n_bins=num_bins, strategy='quantile')
        self.digitizer.fit(self.dataset)

        self.train_size = int(len(self.dataset) * train_ratio)
        self.d_raw = d_raw[self.train_size:]

    def digitize(self):
        self.dataset = self.digitizer.transform(self.dataset)

    def inverse_digitize(self, one_hot):
        data = self.digitizer.inverse_transform(one_hot)
        return data

    def get_train_data(self):
        return self.createTrainData_nparray(self.dataset[:self.train_size], self.seq_length,
                                            self.output_count, self.window_step)

    def get_val_data(self):
        return self.createTrainData_nparray(self.dataset[self.train_size:], self.seq_length,
                                            self.output_count,
                                            self.window_step)

    def get_test_data(self):
        return self.createTrainData_nparray(self.dataset[self.train_size:], self.seq_length,
                                            self.test_output_count, 1)

    def get_raw_data(self):
        return self.createTrainData_nparray(self.d_raw, self.seq_length, self.output_count, 1)

    def get_input_dim(self):
        return [self.seq_length, self.dataset.shape[1]]

    def get_output_dim(self):
        return [self.output_count, self.dataset.shape[1]]


if __name__ == "__main__":
    train_ratio = 0.8
    symbol = ["EURUSD"]
    ohlc = ["close"]
    ema = 5
    oscillator = None
    ema_list = None
    diff = True
    predict_count = 1
    date = datetime(2019, 6, 7)
    n_sample = 10000
    timeframe = MetaTrader5.MT5_TIMEFRAME_M1

    seq_length = 1024
    output_count = 16

    test = FXTMCategorical(train_ratio, seq_length, output_count, symbol, ohlc, date, n_ema=ema,
                           osc_list=oscillator,
                           test_output_count=predict_count,
                           window_step=output_count,
                           num_samples=n_sample,
                           diff=diff,
                           timeframe=timeframe)
    og_dataset = test.dataset
    test.digitize()
    print(test.get_input_dim())
    print(test.get_output_dim())
