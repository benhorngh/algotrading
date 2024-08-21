import logging
from enum import Enum

import pandas as pd

from common import utils, cache_manager, stocks_data
from predictor import (
    prophet_predictor,
    ideal_predictor,
    random_predictor,
    linear_regression_predictor,
    random_forest_predictor,
    last_year_predictor,
    claude_predictor,
    logistic_predictor,
    lstm_predictor,
)


class PredictorOption(str, Enum):
    ideal = "ideal"
    prophet = "prophet"
    random = "random"
    linear_regression = "linear_regression"
    random_forest = "random_forest"
    last_year = "last_year"
    claude = "claude"
    logistic = "logistic"
    lstm = "lstm"


option_to_predictor = {
    PredictorOption.ideal: ideal_predictor,
    PredictorOption.prophet: prophet_predictor,
    PredictorOption.random: random_predictor,
    PredictorOption.linear_regression: linear_regression_predictor,
    PredictorOption.random_forest: random_forest_predictor,
    PredictorOption.last_year: last_year_predictor,
    PredictorOption.claude: claude_predictor,
    PredictorOption.logistic: logistic_predictor,
    PredictorOption.lstm: lstm_predictor,
}


def create_prediction(
    run_id: str, stocks: pd.DataFrame, hold_days: int, predictor_option: PredictorOption
):
    logging.info("Creating prediction")
    max_day = utils.get_stock_last_day(stocks)
    future_trading_dates = utils.get_future_trading_dates(max_day, hold_days)

    predictions = pd.DataFrame(index=future_trading_dates)
    predictions.index.name = stocks_data.DATE_COL

    predictor = option_to_predictor[predictor_option]
    if hasattr(predictor, "pre_predict"):
        predictor.pre_predict(stocks, hold_days)

    for symbol in stocks.columns:
        prediction = predictor.predict(stocks[[symbol]], hold_days)
        predictions[symbol] = prediction
    cache_manager.save_run_file(run_id, cache_manager.CacheFile.prediction, predictions)


def get_prediction(run_id: str) -> pd.DataFrame:
    return cache_manager.get_run_file(
        run_id, cache_manager.CacheFile.prediction, index_col=stocks_data.DATE_COL
    )
