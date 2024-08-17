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
        predictor_option=PredictorOption.lstm,
        symbols=SYMBOLS[:10],
        max_number_of_stocks_to_buy=3,
    )


if __name__ == "__main__":
    main()
