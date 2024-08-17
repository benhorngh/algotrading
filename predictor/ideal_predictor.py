import pandas as pd

from common import utils, stocks_data


def predict(stock: pd.DataFrame, days: int) -> list[float]:
    max_day = utils.get_stock_last_day(stock)
    future_trading_dates = utils.get_future_trading_dates(max_day, days)
    stock_symbol = stock.columns[0]
    stock_actual_data = stocks_data.get_global_stocks([stock_symbol])
    future = [stock_actual_data[stock_symbol][str(d)] for d in future_trading_dates]
    return future
