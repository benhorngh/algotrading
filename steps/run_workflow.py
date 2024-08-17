import logging

from steps import config_step, evaluate_step, prediction_step, strategy_step
from common import logger_setup, stocks_data
from steps.config_step import RunConfig
from common.stocks import SYMBOLS

from datetime import date, datetime

logger_setup.init_logger()


def generate_run_id():
    return datetime.utcnow().isoformat()


def run(
    predictor_option: prediction_step.PredictorOption = None,
    symbols: list[str] = None,
    cutoff_date: date = None,
    hold_days: int = 5,
    max_number_of_stocks_to_buy: int = 5,
) -> str:
    # config
    if predictor_option is None:
        predictor_option = prediction_step.PredictorOption.prophet
    if symbols is None:
        symbols = SYMBOLS[:5]
    if cutoff_date is None:
        cutoff_date = date(year=2024, month=8, day=1)
    config = RunConfig(
        run_id=generate_run_id(),
        predictor_option=predictor_option,
        cutoff_date=cutoff_date,
        symbols=symbols,
        hold_days=hold_days,
        max_number_of_stocks_to_buy=max_number_of_stocks_to_buy,
    )
    config_step.save_config(config)
    logging.info(
        f"Running predictor {predictor_option}. Run id: {config.run_id}. Cutoff: {config.cutoff_date}"
    )

    # stocks
    stocks_data.create_run_stocks(config.run_id, config.cutoff_date, config.symbols)
    stocks = stocks_data.get_run_stocks(config.run_id)

    # prediction
    prediction_step.create_prediction(
        config.run_id, stocks, config.hold_days, config.predictor_option
    )
    predictions = prediction_step.get_prediction(config.run_id)

    # select symbols
    strategy_step.create_strategy(
        config.run_id, predictions, max_number_of_stocks_to_buy
    )
    symbols_to_invest = strategy_step.get_run_strategy(config.run_id)

    # eval
    evaluate_step.evaluate_strategy(
        config.run_id, config.cutoff_date, symbols_to_invest, config.hold_days
    )
    return config.run_id
