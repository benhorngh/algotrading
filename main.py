from datetime import date

from steps import run_workflow

from steps.prediction_step import PredictorOption
from common import stocks_data
from common.stocks import SYMBOLS


def setup():
    print("setup...")
    stocks_data.setup_stocks(SYMBOLS)


def main():
    # setup()
    run_workflow.run(
        predictor_option=PredictorOption.random_forest,
        symbols=SYMBOLS,
        max_number_of_stocks_to_buy=3,
        hold_days=30,
        cutoff_date=date(year=2024, month=5, day=1),
    )


if __name__ == "__main__":
    main()
