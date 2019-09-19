from MetaTrader5 import *
from datetime import datetime
import pandas as pd
from pandas.plotting import register_matplotlib_converters
import matplotlib.pyplot as plt

register_matplotlib_converters()


class MT5DataGetter:
    def __init__(self, symbols):
        self.symbols = symbols
        self.tick_column_names = ['time', 'bid', 'ask', 'last', 'volume', 'flags']
        self.candle_column_names = ['time', 'open', 'high', 'low', 'close', 'tick_volume', 'spread', 'real_volume']

    def gettickdata(self, last_date, how_many, interval='10S'):
        symbols_tickdata = []

        MT5Initialize()
        MT5WaitForTerminal()

        for symbol in self.symbols:
            tick = MT5CopyTicksFrom(symbol, last_date, how_many, MT5_COPY_TICKS_ALL)
            tick = pd.DataFrame(list(tick), columns=self.tick_column_names)
            tick = tick.set_index(['time'])
            tick = tick.resample(interval).mean()
            tick.pop('last')
            tick.pop('volume')
            tick = tick.interpolate(limit=10)
            symbols_tickdata.append(tick)

        MT5Shutdown()

        return symbols_tickdata

    def getcandledata(self, last_date, how_many, interval_MT5):
        symbols_candledata = []

        MT5Initialize()
        MT5WaitForTerminal()

        for symbol in self.symbols:
            candle = MT5CopyRatesFrom(symbol, interval_MT5, last_date, how_many)
            candle = pd.DataFrame(list(candle), columns=self.candle_column_names)
            candle = candle.set_index(['time'])
            symbols_candledata.append(candle)

        MT5Shutdown()

        return symbols_candledata

    def getmostrecent(self, how_many, interval_MT5):
        symbols_candledata = []

        MT5Initialize()
        MT5WaitForTerminal()

        for symbol in self.symbols:
            candle = MT5CopyRatesFromPos(symbol, interval_MT5, 0, how_many)
            candle = pd.DataFrame(list(candle), columns=self.candle_column_names)
            candle = candle.set_index(['time'])
            symbols_candledata.append(candle)

        MT5Shutdown()

        return symbols_candledata

if __name__ == "__main__":
    symbol_list = ["EURUSD", "USDJPY", "GBPUSD", "USDCHF"]

    test = MT5DataGetter(symbol_list)
    candles_m1 = test.getcandledata(datetime(2019, 4, 20), 80000, MT5_TIMEFRAME_M1)

    eurusd1m = candles_m1[0]
    plt.plot(eurusd1m['close'])
    plt.show()

