import logging
from datetime import date

import pandas as pd

from common import utils, cache_manager, stocks_data, consts


def evaluate_strategy(
    run_id: str, cutoff_date: date, symbols: list[str], hold_days: int
) -> float:
    logging.info(f"Evaluating symbols {symbols}")
    future_trading_dates = utils.get_future_trading_dates(cutoff_date, hold_days)
    stocks_actual_data = stocks_data.get_global_stocks(symbols)
    buy_date = future_trading_dates[0]
    sell_date = future_trading_dates[-1]
    total_buy = 0
    total_profit = 0
    symbol_to_percent = {}
    for symbol in symbols:
        buy_price = stocks_actual_data[symbol][str(buy_date)]
        sell_price = stocks_actual_data[symbol][str(sell_date)]
        profit = sell_price - buy_price
        profit = profit - consts.IBKR_INVESTMENT_FEE
        profit_percent = profit / buy_price * 100
        symbol_to_percent[symbol] = profit_percent
        total_buy += buy_price
        total_profit += profit
    save_symbols_actual_profit(run_id, symbol_to_percent)
    if total_buy != 0:
        total_profit_percent = total_profit / total_buy * 100
    else:
        total_profit_percent = 0
    save_total_profit(
        run_id,
        total_spent=total_buy,
        number_of_stocks=len(symbols),
        total_profit=total_profit,
        percent=total_profit_percent,
    )

    logging.info(f"Total spent: { total_buy}")
    logging.info(f"Total profit: {total_profit}")
    logging.info(f"Total profit percent: {total_profit_percent}")
    return total_profit_percent


def save_symbols_actual_profit(run_id: str, symbols_to_percent: dict[str, float]):
    df = utils.create_simple_df(symbols_to_percent, "symbol", "percent")
    cache_manager.save_run_file(
        run_id, cache_manager.CacheFile.actual_profit_per_symbol, df, with_index=False
    )


def save_total_profit(
    run_id: str,
    total_spent: float,
    number_of_stocks: int,
    total_profit: float,
    percent: float,
):
    df = pd.DataFrame(
        data={
            "total_spent": [total_spent],
            "number_of_stocks": [number_of_stocks],
            "total_profit": [total_profit],
            "percent": [percent],
        }
    )
    cache_manager.save_run_file(
        run_id, cache_manager.CacheFile.actual_profit_total, df, with_index=False
    )


def get_run_total_percent(run_id: str) -> float:
    return cache_manager.get_run_file(
        run_id, cache_manager.CacheFile.actual_profit_total
    )["percent"].values[0]
