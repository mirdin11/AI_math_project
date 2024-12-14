import numpy as np
import pandas as pd

from datetime import datetime
from arch import arch_model

from .stock_data import StockData
from .utils import Utils


class StockStatistics:

    @staticmethod
    def calculate_return(stock_data: StockData) -> pd.Series:

        return stock_data.data["close"].pct_change().dropna()

    @staticmethod
    def calculate_overall_return(stock_data: StockData) -> np.float64:

        return stock_data.data["close"].iloc[-1] / stock_data.data["close"].iloc[0] - 1

    @staticmethod
    def calculate_mean_return(stock_data: StockData) -> np.float64:

        return np.mean(StockStatistics.calculate_return(stock_data=stock_data))

    @staticmethod
    def calculate_simple_volatility(stock_data: StockData) -> np.float64:

        return np.std(StockStatistics.calculate_return(stock_data=stock_data))

    @staticmethod
    def calculate_volatility_using_GARCH(stock_data: StockData) -> np.float64:

        log_return = np.log(
            stock_data.data["close"] / stock_data.data["close"].shift(1)
        ).dropna()
        model = arch_model(log_return, vol="Garch", p=1, q=1, rescale=False)
        model_fit = model.fit(disp="off")

        return np.mean(model_fit.conditional_volatility)

    @staticmethod
    def calculate_correlation(
        stock_data1: StockData, stock_data2: StockData
    ) -> np.float64:

        curr_date = datetime.now()
        first_date = Utils.find_closer_date(
            date1=stock_data1.data["date"].iloc[0],
            date2=stock_data2.data["date"].iloc[0],
            ref=curr_date,
        )

        trimmed_stock_data1 = StockData(
            stock_data1.ticker, stock_data1.data[stock_data1.data["date"] >= first_date]
        )
        trimmed_stock_data2 = StockData(
            stock_data2.ticker, stock_data2.data[stock_data2.data["date"] >= first_date]
        )

        if len(trimmed_stock_data1.data["close"]) == 2:
            print(stock_data1.ticker, stock_data2.ticker)

        ret1 = StockStatistics.calculate_return(stock_data=trimmed_stock_data1)
        ret2 = StockStatistics.calculate_return(stock_data=trimmed_stock_data2)

        return np.corrcoef(ret1, ret2)[0, 1]
