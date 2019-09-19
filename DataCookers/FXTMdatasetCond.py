from __future__ import absolute_import, division, print_function

import os

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

# TensorFlow and tf.keras
import matplotlib.pyplot as plt
# Helper libraries
import numpy as np
from scipy import stats
from DataCookers.MT5DataGetter import MT5DataGetter
from datetime import datetime
import MetaTrader5
from sklearn import preprocessing


class CondMarketDataGenerator(object):

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
    def createTrainData_nparray(data, seqLength, predLength=1):
        i = 0
        dataX = []
        dataY = []
        while (i < (len(data) - seqLength - predLength)):
            dataX.append(data[i:i + seqLength])
            dataY.append(data[i + seqLength:(i + seqLength + predLength)])
            i += 1

        return np.array(dataX), np.array(dataY)

    @staticmethod
    def normalize(data):
        numerator = data - np.min(data, 0)
        denominator = np.max(data, 0) - np.min(data, 0)
        return numerator / (denominator + 1e-7)

    def normalize_per_symbol(self, data):
        dataset = []
        for i in range(len(data[0])):
            temp = data[:, i]
            temp = self.normalize(temp)
            dataset.append(temp)
        dataset = np.stack(dataset, axis=1);
        return dataset

    @staticmethod
    def standardize(data):
        m = np.mean(data, axis=0)
        stdev = np.std(data, axis=0)
        return (data - m) / stdev

    @staticmethod
    def deStandardize(prevData, currentData):
        m = np.mean(prevData, axis=0)
        stdev = np.std(prevData, axis=0)
        return currentData * stdev + m

    @staticmethod
    def DeNormalize(prevData, currentData):
        min = np.min(prevData, 0)
        denominator = np.max(prevData, 0) - np.min(prevData, 0)
        return currentData * denominator + min

    @staticmethod
    def getMinTimeStep(data):
        min = data[0].shape[0]
        for i in range(len(data)):
            if (min > data[i].shape[0]):
                min = data[i].shape[0]
        return min

    @staticmethod
    def get_delta(Y):
        Y_shiftright = np.concatenate(([Y[0]], Y), axis=0)
        Y_shiftright = np.delete(Y_shiftright, len(Y) - 1, axis=0)
        return np.subtract(Y_shiftright, Y)

    @staticmethod
    def moving_avg(Y, timestep):
        Y_new = []
        for i in range(len(Y) - timestep):
            Y_chunk = Y[i:i + timestep]
            mean = np.mean(Y_chunk, axis=0)
            Y_new.append(mean)
        return np.stack(Y_new)

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
    def remove_outliers(data, threshold=7):
        z = np.abs(stats.zscore(data))
        points = np.where(z > threshold)
        xs = points[0]
        ys = points[1]
        for i in range(len(points[0])):
            x = xs[i]
            y = ys[i]
            data[x, y] = np.mean(data[x - 20:x, y])

    def __init__(self, train_ratio,
                 seq_length,
                 output_count,
                 symbol_list,
                 last_date,
                 num_samples=99999,
                 mostrecent=False,
                 timeframe=MetaTrader5.MT5_TIMEFRAME_M1):

        if mostrecent:
            raw_datasets = MT5DataGetter(symbol_list).getmostrecent(num_samples, timeframe)
        else:
            raw_datasets = MT5DataGetter(symbol_list).getcandledata(last_date, num_samples, timeframe)

        dataset = []

        for raw_dataset in raw_datasets:
            close = np.expand_dims(raw_dataset['close'].values, axis=-1)
            dataset.append(close)

        dataset = np.stack(dataset, axis=1)  # (timesteps, markets, features)
        d_raw = np.copy(dataset)
        d_nodiff = np.copy(dataset)[:-1]
        dataset = self.exp_moving_avg(dataset, 10)
        d_nodiff_ema = self.exp_moving_avg(d_nodiff, 20)

        # max = 6
        # d = [self.exp_moving_avg(dataset, 20*(2**i))[-(dataset.shape[0]-20*(2**(max-1))):] for i in range(max)]
        # d = np.concatenate(d, axis=-1)
        # dataset = dataset[-(dataset.shape[0]-20*(2**(max-1))):]
        # print(d.shape)

        dataset = np.diff(dataset, axis=0, prepend=dataset[0:1])
        # d = np.diff(d, axis=0)

        shape = dataset.shape
        d_shape = d_nodiff.shape
        dataset = np.reshape(dataset, [dataset.shape[0], dataset.shape[1] * dataset.shape[2]])
        # d_nodiff = np.reshape(d_nodiff, [d_nodiff.shape[0], d_nodiff.shape[1]*d_nodiff.shape[2]])
        # d = np.reshape(d, [d.shape[0], d.shape[1]*d.shape[2]])
        # self.remove_outliers(dataset)
        prev_dataset = dataset
        self.stdizer = preprocessing.StandardScaler().fit(prev_dataset)
        dataset = self.stdizer.transform(dataset)

        plt.plot(dataset)
        plt.show()

        self.og_dataset = dataset

        dataset = np.reshape(dataset, [shape[0], shape[1], shape[2]])
        d_nodiff = np.reshape(d_nodiff, [d_shape[0], d_shape[1], d_shape[2]])

        train_size = int(len(dataset) * train_ratio)
        train_dataset = dataset[:train_size]
        d_raw_tst = d_raw[train_size:]
        # d_nodiff = d_nodiff[train_size:]
        d_nodiff_ema = d_nodiff_ema[train_size:]
        test_dataset = dataset[train_size:]

        self.trainX, self.trainY = self.createTestData_nparray(train_dataset, seq_length, output_count)

        self.validX, self.validY = self.createTestData_nparray(test_dataset, seq_length, output_count)

        self.rawX, self.rawY = self.createTrainData_nparray(d_raw_tst, seq_length, output_count)

        self.nodifX, self.nodifY = self.createTrainData_nparray(d_nodiff, seq_length, output_count)

        self.nodif_emaX, self.nodif_emaY = self.createTrainData_nparray(d_nodiff_ema, seq_length, output_count)

        self.testX, self.testY = self.createTrainData_nparray(test_dataset, seq_length, output_count)

        self.input_dim = self.trainX.shape[1:]  # dimension of inputs
        self.output_dim = self.trainY.shape[1:]


if __name__ == "__main__":
    symbol = ["EURUSD"]
    test = CondMarketDataGenerator(0.8, 1024, 256, symbol, datetime(2019, 5, 15), num_samples=10000,
                                   timeframe=MetaTrader5.MT5_TIMEFRAME_M15)

    # plt.plot(test.nodifX[0].squeeze())
    # plt.plot(test.rawX[0].squeeze())
    # plt.show()
