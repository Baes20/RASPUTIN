from __future__ import absolute_import, division, print_function

import os

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

# TensorFlow and tf.keras
import tensorflow as tf
import random
import pandas as pd

# Helper libraries
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf


class NasdaqGenerator(object):

    def createTestData_nparray(self, data, seqLength, predLength=1):
        i = 0
        dataX = []
        dataY = []
        while (i < (len(data) - seqLength - predLength)):
            dataX.append(data[i:i + seqLength])
            dataY.append(data[i + seqLength:(i + seqLength + predLength)])
            i += predLength

        return np.array(dataX), np.array(dataY)

    def createTrainData_nparray(self, data, seqLength, predLength=1):
        i = 0
        dataX = []
        dataY = []
        while (i < (len(data) - seqLength - predLength)):
            dataX.append(data[i:i + seqLength])
            dataY.append(data[i + seqLength:(i + seqLength + predLength)])
            i += 1

        return np.array(dataX), np.array(dataY)

    def normalize(self, data):
        numerator = data - np.min(data, 0)
        denominator = np.max(data, 0) - np.min(data, 0)
        return numerator / (denominator + 1e-7)

    def standardize(self, data):
        m = np.mean(data)
        stdev = np.std(data)
        return (data - m) / stdev

    def deStandardize(self, prevData, currentData):
        m = np.mean(prevData)
        stdev = np.std(prevData)
        return currentData * stdev + m

    def DeNormalize(self, prevData, currentData):
        min = np.min(prevData, 0)
        denominator = np.max(prevData, 0) - np.min(prevData, 0)
        return currentData * denominator + min

    def getMinTimeStep(self, data):
        min = data[0].shape[0]
        for i in range(len(data)):
            if (min > data[i].shape[0]):
                min = data[i].shape[0]
        return min

    def get_delta(self, Y):
        Y_shiftright = np.concatenate(([Y[0]], Y), axis=0)
        Y_shiftright = np.delete(Y_shiftright, len(Y) - 1, axis=0)
        return np.subtract(Y_shiftright, Y)

    def __init__(self, train_ratio, seq_length, output_count, batch_size):

        nasdaq100_small_raw = pd.read_csv(
            filepath_or_buffer="D:/Projects/tensor2/NASDAQ100/nasdaq100/small/nasdaq100_padding.csv")
        dataset = []

        for i in range(len(nasdaq100_small_raw.values[0])):
            temp = nasdaq100_small_raw.values[:, i]
            dataset.append(temp)
        dataset = np.stack(dataset, axis=1)
        # dataset = np.reshape(dataset, [dataset.shape[0], dataset.shape[1], 1])
        print(dataset.shape)
        self.dataset = dataset
        dataset = np.diff(dataset, axis=0)
        plt.plot(dataset[:1000, -1])
        plt.show()
        plot_acf(dataset[:1000, -1])
        plt.show()

        train_size = int(len(dataset) * train_ratio)
        train_dataset = dataset[:train_size]
        test_dataset = dataset[train_size:]

        self.trainX, self.trainY = self.createTrainData_nparray(train_dataset, seq_length, output_count)
        self.testX, self.testY = self.createTestData_nparray(test_dataset, seq_length, output_count)

        self.batch_size = batch_size
        self.input_dim = self.trainX.shape[1:]  # dimension of inputs
        self.output_dim = self.trainY.shape[1:]

if __name__ == "__main__":
    a = NasdaqGenerator(0.8, 64, 8, 16)
