import os
from datetime import datetime
from enum import Enum
from functools import lru_cache

import pandas as pd

CACHE_PATH = ".cache/"
RUN_CACHE_PATH = os.path.join(CACHE_PATH, "run")
TMP_CACHE_PATH = os.path.join(CACHE_PATH, "tmp")


class CacheFile(str, Enum):
    stock_data = "stock_data.csv"
    run_stock_data = "run_stock_data.csv"
    prediction = "prediction.csv"
    run_config = "run_config.csv"
    symbols_to_invest = "symbols_to_invest.csv"
    predicted_profit = "predicted_profit.csv"
    actual_profit_per_symbol = "actual_profit_per_symbol.csv"
    actual_profit_total = "actual_profit_total.csv"
    symbol_prediction_score = "symbol_prediction_score.csv"


def _create_cache_folders():
    if not os.path.exists(CACHE_PATH):
        os.makedirs(CACHE_PATH)
    if not os.path.exists(RUN_CACHE_PATH):
        os.makedirs(RUN_CACHE_PATH)
    if not os.path.exists(TMP_CACHE_PATH):
        os.makedirs(TMP_CACHE_PATH)


def _create_run_folder(run_id: str):
    run_folder_path = os.path.join(RUN_CACHE_PATH, run_id)
    if not os.path.exists(run_folder_path):
        os.makedirs(run_folder_path)


def save_run_file(
    run_id: str, file_name: str, data: pd.DataFrame, with_index: bool = True
):
    _create_run_folder(run_id)
    file_path = os.path.join(RUN_CACHE_PATH, run_id, file_name)
    _save_file(file_path, data, with_index)


def save_global_file(file_name: str, data: pd.DataFrame, with_index: bool = True):
    file_path = os.path.join(CACHE_PATH, file_name)
    _save_file(file_path, data, with_index)


def save_tmp_file(file_name: str, data: pd.DataFrame, with_index: bool = True):
    file_name = f"{datetime.utcnow().isoformat()}-{file_name}"
    file_path = os.path.join(TMP_CACHE_PATH, file_name)
    _save_file(file_path, data, with_index)


def _save_file(file_path: str, data: pd.DataFrame, with_index: bool):
    if with_index:
        data.to_csv(file_path)
    else:
        data.to_csv(file_path, index=False)


@lru_cache
def get_global_file(file_name: str, index_col: str | bool = False) -> pd.DataFrame:
    file_path = os.path.join(CACHE_PATH, file_name)
    return _get_file(file_path, index_col)


def get_run_file(
    run_id: str, file_name: str, index_col: str | bool = False
) -> pd.DataFrame:
    file_path = os.path.join(RUN_CACHE_PATH, run_id, file_name)
    return _get_file(file_path, index_col)


def get_tmp_file(file_name: str, index_col: str | bool = False) -> pd.DataFrame:
    file_path = os.path.join(TMP_CACHE_PATH, file_name)
    return _get_file(file_path, index_col)


@lru_cache
def _get_file(file_path: str, index_col: str | bool = False) -> pd.DataFrame:
    return pd.read_csv(file_path, index_col=index_col)


_create_cache_folders()
