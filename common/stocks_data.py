import logging
from datetime import datetime, timedelta, date
from pprint import pprint

import pandas as pd
import yfinance as yf

from common import cache_manager
from common.stocks import SYMBOLS

DATE_COL = "Date"
ADJ_CLOSE_PRICE_COL = "Adj Close"
CLOSE_PRICE_COL = "Close"


def _download_stocks(symbols: list[str], days: int = 365 * 5):
    today = datetime.today()
    start_date = today - timedelta(days=days)
    data = yf.download(symbols, start=str(start_date.date()), end=str(today.date()))

    closing_data: pd.DataFrame = data[CLOSE_PRICE_COL]
    return closing_data


def setup_stocks(symbols: list[str]):
    stocks = _download_stocks(symbols)
    cache_manager.save_global_file(cache_manager.CacheFile.stock_data, stocks)


def get_current_price(symbol: str) -> float:
    ticker = yf.Ticker(symbol)
    today_data = ticker.history(period="1d")
    return today_data[CLOSE_PRICE_COL].iloc[0]


def filter_by_symbols(df: pd.DataFrame, symbols: list[str]) -> pd.DataFrame:
    return df[symbols]


def filter_by_date(df: pd.DataFrame, cutoff_date: date) -> pd.DataFrame:
    tdf = df[df.index <= str(cutoff_date)]
    return tdf.tail(365 * 3)
    # return tdf


def create_run_stocks(run_id: str, cutoff_date: date, symbols: list[str]):
    logging.info("Creating run stocks")
    stocks = cache_manager.get_global_file(
        cache_manager.CacheFile.stock_data, index_col=DATE_COL
    )
    stocks = filter_by_symbols(stocks, symbols)
    stocks = filter_by_date(stocks, cutoff_date)
    cache_manager.save_run_file(run_id, cache_manager.CacheFile.run_stock_data, stocks)


def get_run_stocks(run_id: str) -> pd.DataFrame:
    return cache_manager.get_run_file(
        run_id, cache_manager.CacheFile.run_stock_data, index_col=DATE_COL
    )


def get_global_stocks(symbols: list[str]) -> pd.DataFrame:
    stocks = cache_manager.get_global_file(
        cache_manager.CacheFile.stock_data, index_col=DATE_COL
    )
    stocks = filter_by_symbols(stocks, symbols)
    return stocks


def _get_current_prices(symbols: list[str]):
    symbol_to_price = {}
    for symbol in symbols:
        try:
            symbol_to_price[symbol] = get_current_price(symbol)
        except:
            print("error", symbol)
            symbol_to_price[symbol] = None
    pprint(symbol_to_price)
    return symbol_to_price


if __name__ == "__main__":
    ...
    # save_stocks(SYMBOLS[:3])
    # print(filter_by_symbols(read_cache_file(), ['ABT']))
    # d = date(year=2023, month=8,day=6)
    # print(filter_by_date(read_cache_file(), d))
    ...
    # setup_stocks(SYMBOLS[:10])
    # d = date(year=2023, month=8, day=6)
    # get_run_stocks("a123", d, SYMBOLS[:1])
    _get_current_prices(SYMBOLS)
