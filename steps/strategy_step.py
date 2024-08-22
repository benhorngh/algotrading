import logging

import pandas as pd

from common import utils, cache_manager, consts

MINIMUM_RETURN = 0.7


def create_strategy(run_id: str, prediction: pd.DataFrame, max_number_of_stocks: int):
    logging.info("Creating strategy")
    symbol_to_profit_percent = {}
    for symbol in prediction.columns:
        future = prediction[symbol]
        buy_price = future.iloc[0]
        sell_price = future.iloc[-1]
        profit = sell_price - buy_price
        profit = profit - consts.IBKR_INVESTMENT_FEE
        symbol_to_profit_percent[symbol] = profit / buy_price * 100

    save_symbols_expected_profit(run_id, symbol_to_profit_percent)
    symbols = _select_symbols(symbol_to_profit_percent, max_number_of_stocks)
    symbols_df = pd.DataFrame(data={"symbol": symbols})
    cache_manager.save_run_file(
        run_id, cache_manager.CacheFile.symbols_to_invest, symbols_df, with_index=False
    )


def get_run_strategy(run_id: str) -> list[str]:
    return cache_manager.get_run_file(
        run_id, cache_manager.CacheFile.symbols_to_invest
    )["symbol"].values


def _select_symbols(d: dict[str, float], max_number_of_stocks: int) -> list[str]:
    top_keys = sorted(d.items(), key=lambda item: item[1], reverse=True)[
        :max_number_of_stocks
    ]
    top_keys = [kv[0] for kv in top_keys if kv[1] > MINIMUM_RETURN]
    return top_keys


def save_symbols_expected_profit(run_id: str, symbols_to_percent: dict[str, float]):
    df = utils.create_simple_df(symbols_to_percent, "symbol", "percent")
    cache_manager.save_run_file(
        run_id, cache_manager.CacheFile.predicted_profit, df, with_index=False
    )
