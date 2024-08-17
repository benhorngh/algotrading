from datetime import timedelta

import pandas as pd

from common import utils


def predict(stock: pd.DataFrame, days: int) -> list[float]:
    symbol = stock.columns[0]
    max_day = utils.get_stock_last_day(stock)
    last_year = max_day - timedelta(days=365)

    future_trading_dates = utils.get_future_trading_dates(last_year, days)
    last_year_data = stock.loc[[str(f) for f in future_trading_dates]]

    last_year_price = None
    for i in range(10):
        if str(last_year) in stock.index:
            last_year_price = stock.loc[str(last_year)].values[0]
            break
        else:
            last_year = last_year - timedelta(days=1)

    last_year_percent = last_year_data[symbol] / last_year_price * 100
    percent = last_year_percent.values
    current_value = stock.loc[str(max_day)].values[0]
    return [p / 100 * current_value for p in percent]
