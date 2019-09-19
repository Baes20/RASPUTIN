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
from sklearn import preprocessing
import MetaTrader5
from AI.FXTM_Summarizer import WaveVAE


class MarketDataGenerator(object):

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

    @staticmethod
    def reshape4Dto2D(data):
        return np.reshape(data, (data.shape[0] * data.shape[1], data.shape[2] * data.shape[3]))

    @staticmethod
    def reshape2Dto4D(original_shape, data):
        return np.reshape(data, (original_shape[0], original_shape[1], original_shape[2], original_shape[3]))

    def __init__(self, train_ratio,
                 seq_length,
                 output_count,
                 symbol_list,
                 last_date,
                 num_samples=99999,
                 mostrecent=False,
                 testing=False,
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

        shape = dataset.shape
        self.input_dim = [seq_length, shape[1], shape[2]]
        self.output_dim = [output_count, shape[1], shape[2]]

        # dataset = self.exp_moving_avg(dataset, 10)
        # dataset = np.diff(dataset, axis=0, append=dataset[-1:])
        dataset = np.reshape(dataset, [dataset.shape[0], dataset.shape[1] * dataset.shape[2]])
        prev_dataset = dataset

        scaler = preprocessing.StandardScaler().fit(prev_dataset)
        dataset = scaler.transform(dataset)

        dataset = np.reshape(dataset, [shape[0], shape[1], shape[2]])

        VAE = WaveVAE.WaveVAE(200, 0.0001)
        VAE.compile([seq_length * 2, shape[1], shape[2]], 'adam')
        VAE.VAE.load_weights('./SavedModel/VAE/WaveVAE.h5')

        dataset = np.stack([dataset[2048 * i:2048 * (i + 1)] for i in range(int(len(dataset) / 2048))])
        dataset = VAE.VAE.predict(dataset)

        dataset = np.reshape(dataset, (dataset.shape[0] * dataset.shape[1], dataset.shape[2] * dataset.shape[3]))

        dataset = scaler.inverse_transform(dataset)

        summarized = np.copy(dataset)
        summarized = np.reshape(summarized, [shape[0], shape[1], shape[2]])

        plt.plot(np.squeeze(d_raw))
        plt.plot(np.squeeze(summarized))
        plt.show()

        dataset = np.diff(dataset, axis=0, prepend=dataset[0:1])

        self.stdizer = preprocessing.StandardScaler().fit(dataset)

        dataset = self.stdizer.transform(dataset)

        self.dataset = np.reshape(dataset, [shape[0], shape[1], shape[2]])

        train_size = int(len(self.dataset) * train_ratio)

        train_dataset = self.dataset[:train_size]
        test_dataset = self.dataset[train_size:]
        d_raw = d_raw[train_size:]
        summarized = summarized[train_size:]

        self.trainX, self.trainY = self.createTestData_nparray(train_dataset, seq_length, output_count)

        self.validX, self.validY = self.createTestData_nparray(test_dataset, seq_length, output_count)

        self.rawX, self.rawY = self.createTrainData_nparray(d_raw, seq_length * 2, output_count * 2)

        # self.vaeX, self.vaeY = self.createTrainData_nparray(summarized, seq_length, output_count)

        # self.testX, self.testY = self.createTrainData_nparray(test_dataset, seq_length, output_count)

        og_shape = self.rawX.shape
        self.vaeX = []
        self.vaeY = []
        self.testX = []
        self.testY = []

        if testing:
            shape = self.rawX.shape

            tempX = self.reshape4Dto2D(self.rawX)
            tempX = scaler.transform(tempX)
            tempX = self.reshape2Dto4D(shape, tempX)

            tempX = VAE.VAE.predict(tempX)

            tempX = self.reshape4Dto2D(tempX)
            tempX = scaler.inverse_transform(tempX)
            tempX = self.reshape2Dto4D(shape, tempX)

            testX = np.diff(tempX, axis=1, prepend=tempX[:, 0:1])
            tempXlast = tempX[:, -1:]

            testX = self.reshape4Dto2D(testX)
            testX = self.stdizer.transform(testX)
            testX = self.reshape2Dto4D(shape, testX)

            self.vaeX = tempX
            self.testX = testX

            shape2 = self.rawY.shape

            tempY = self.reshape4Dto2D(self.rawY)
            tempY = scaler.transform(tempY)
            tempY = self.reshape2Dto4D(shape2, tempY)

            tempY = VAE.VAE.predict(tempY)

            tempY = self.reshape4Dto2D(tempY)
            tempY = scaler.inverse_transform(tempY)
            tempY = self.reshape2Dto4D(shape2, tempY)

            testY = np.diff(tempY, axis=1, prepend=tempXlast)

            testY = self.reshape4Dto2D(testY)
            testY = self.stdizer.transform(testY)
            testY = self.reshape2Dto4D(shape2, testY)

            self.vaeY = tempY
            self.testY = testY

        # if testing:
        #     for i in range(len(self.rawX)):
        #         tempX = scaler.transform(np.squeeze(self.rawX[i:i + 1]))
        #         tempX = np.expand_dims(np.expand_dims(tempX, axis=0), axis=-1)
        #         tempX = VAE.VAE.predict(tempX)
        #         tempX = scaler.inverse_transform(np.squeeze(tempX))
        #         testX = np.diff(tempX, axis=0, prepend=tempX[0:1])
        #         tempXlast = tempX[-1:]
        #         testX = self.stdizer.transform(testX)
        #         testX = np.expand_dims(np.expand_dims(testX, axis=0), axis=-1)
        #         tempX = np.expand_dims(np.expand_dims(tempX, axis=0), axis=-1)
        #         self.vaeX.append(tempX)
        #         self.testX.append(testX)
        #
        #         tempY = scaler.transform(np.squeeze(self.rawY[i:i + 1]))
        #         tempY = np.expand_dims(np.expand_dims(tempY, axis=0), axis=-1)
        #         tempY = VAE.VAE.predict(tempY)
        #         tempY = scaler.inverse_transform(np.squeeze(tempY))
        #         testY = np.diff(tempY, axis=0, prepend=tempXlast)
        #         testY = self.stdizer.transform(testY)
        #         testY = np.expand_dims(np.expand_dims(testY, axis=0), axis=-1)
        #         tempY = np.expand_dims(np.expand_dims(tempY, axis=0), axis=-1)
        #         self.vaeY.append(tempY)
        #         self.testY.append(testY)
        #
        #     self.vaeX = np.concatenate(self.vaeX, axis=0)
        #     self.vaeY = np.concatenate(self.vaeY, axis=0)
        #     self.testX = np.concatenate(self.testX, axis=0)
        #     self.testY = np.concatenate(self.testY, axis=0)

        # dimension of inputs


if __name__ == "__main__":
    symbol = ["EURUSD", "EURGBP", "GBPUSD"]
    test = MarketDataGenerator(0.8, 1024, 1024, symbol, datetime(2019, 5, 15), num_samples=16384,
                               timeframe=MetaTrader5.MT5_TIMEFRAME_M15)
