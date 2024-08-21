import pandas as pd

from common import stocks_data
from predictor import predictor_utils


def predict(stock: pd.DataFrame, hold_days: int) -> list[float]:
    stock_symbol = predictor_utils.get_symbol(stock)
    future_trading_dates = predictor_utils.get_future_trading_days(stock, hold_days)
    stock_actual_data = stocks_data.get_global_stocks([stock_symbol])
    future = [stock_actual_data[stock_symbol][str(d)] for d in future_trading_dates]
    return future
