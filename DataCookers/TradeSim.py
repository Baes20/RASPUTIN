from DataCookers.VAEdataset import VaeGen
import os
import MetaTrader5
from datetime import datetime

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
os.chdir("D:/Projects/AI")


class FXTMTradeSimulator:
    def __init__(self, seq_length, symbol, date, n_sample, timeframe, initial_money=10000):
        self.balance = initial_money
        gen = VaeGen(0.8, seq_length, 1, symbol, ["close", "high", "low"], date, test_output_count=1, window_step=1,
                     num_samples=n_sample, n_ema=1, ema_list=None, diff=False, logtransform=False,
                     timeframe=timeframe, preprocess=False)
        self.X, self.Y = gen.get_test_data()
        self.buy_info = {"entryprice": None, "lot": 0}
        self.sell_info = {"entryprice": None, "lot": 0}
        print(self.X.shape, self.Y.shape)
        self.t = 0  # this value indicates the "present" in this simulation
        self.win_count = 0
        self.total_trade = 0

    def get_current_candle(self):
        return self.X[self.t, -1, 0]

    def getX(self):
        return self.X[self.t]

    def getY(self):  # get the next candle(future)
        return self.Y[self.t]

    def update(self, exit_profit_threshold):  # profit in points that automatically exits buy/sell position once exeeded
        future_candle = self.getY()[0, 0]  # c, h, l
        if self.buy_info["entryprice"] is not None:  # buy position process start
            target_price = self.buy_info["entryprice"] + exit_profit_threshold
            if future_candle[2] < target_price < future_candle[1]:  # if next_low < target < next_high
                self.buy_exit(target_price)  # prediction was successful, granting profit
                self.win_count += 1
            else:  # otherwise
                self.buy_exit(future_candle[0])  # automatically exit at the next close price
            self.total_trade += 1
        if self.sell_info["entryprice"] is not None:  # sell position process start
            # c, h, l
            target_price = self.sell_info["entryprice"] - exit_profit_threshold
            if future_candle[2] < target_price < future_candle[1]:  # if next_low < target < next_high
                self.sell_exit(target_price)  # prediction was successful, granting profit
                self.win_count += 1
            else:  # otherwise
                self.sell_exit(future_candle[0])  # automatically exit at the next close price
            self.total_trade += 1

        self.t += 1

    def buy_entry(self, lot):  # buy lot amount at the close price of the current candle
        self.buy_info["entryprice"] = self.get_current_candle()[0]  # entry = close price of current candle
        self.buy_info["lot"] = lot

    def sell_entry(self, lot):
        self.sell_info["entryprice"] = self.get_current_candle()[0]  # entry = close price of current candle
        self.sell_info["lot"] = lot

    def buy_exit(self, exit_price):
        self.balance += (exit_price - self.buy_info["entryprice"]) * self.buy_info["lot"] * (
                1 / 0.00001)  # profit = (pointdiff)*(lot)*
        self.buy_info["entryprice"] = None  # reset entryprice after exit
        self.buy_info["lot"] = 0  # reset lot after exit

    def sell_exit(self, exit_price):
        self.balance += (self.sell_info["entryprice"] - exit_price) * self.sell_info["lot"] * (
                1 / 0.00001)  # profit = (pointdiff)*(lot)
        self.sell_info["entryprice"] = None  # reset entryprice after exit
        self.sell_info["lot"] = 0  # reset lot after exit

    def isBankrupt(self):
        return self.balance <= 0

    def getWinRate(self):
        if self.total_trade > 0:
            return self.win_count / self.total_trade

    def getBalance(self):
        return self.balance

    def summary(self):
        print("time:", self.t, "balance:", self.balance, "win_count:", self.win_count, "total_trade", self.total_trade,
              "winrate:", self.getWinRate(), "trade_frequency:", self.total_trade / self.t)


if __name__ == "__main__":
    seq_length = 32
    train_ratio = 0.8
    symbol = ["EURUSD"]
    ohlc = ["close", "high", "low"]
    date = datetime(2019, 8, 9)
    n_sample = 92160
    timeframe = MetaTrader5.MT5_TIMEFRAME_M1

    FXTMTradeSimulator(seq_length, symbol, ohlc, date, n_sample, timeframe)
