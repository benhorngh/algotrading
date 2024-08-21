from datetime import date

import pandas as pd

from common import utils


def get_last_day(stock: pd.DataFrame) -> date:
    return date.fromisoformat(max(stock.index))


def get_future_trading_days(stock: pd.DataFrame, hold_days: int) -> list[date]:
    last_day = get_last_day(stock)
    future_trading_dates = utils.get_future_trading_dates(last_day, hold_days)
    return future_trading_dates


def get_symbol(stock: pd.DataFrame) -> str:
    return stock.columns[0]


def get_last_price(stock: pd.DataFrame) -> float:
    last_day = get_last_day(stock)
    return stock.loc[str(last_day)].values[0]


def get_prices(stock: pd.DataFrame) -> list[float]:
    return stock[get_symbol(stock)].to_list()
