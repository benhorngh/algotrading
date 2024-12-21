from datetime import timedelta, date
from pprint import pprint

from common import stocks, investment_style
from common.investment_style import InvestmentStyle
from common.logger_setup import init_logger
from common.stocks import SYMBOLS, SYMBOLS_50_150, SYMBOLS_150_300
from steps import evaluate_step, run_workflow
from steps.prediction_step import PredictorOption


def calculate_predictor_score(
    symbols: list[str],
    predictor_option: PredictorOption,
    number_of_tests: int,
    style: InvestmentStyle,
    max_number_of_stocks_to_buy: int = 3,
):
    end_date = style.end_date
    hold_days = style.hold_days
    iteration_to_percent = {}
    for i in range(number_of_tests):
        predictor_run_id = run_workflow.run(
            cutoff_date=end_date,
            predictor_option=predictor_option,
            symbols=symbols,
            max_number_of_stocks_to_buy=max_number_of_stocks_to_buy,
            hold_days=hold_days,
        )
        end_date = end_date - timedelta(days=style.days_delta)

        total_percent = evaluate_step.get_run_total_percent(predictor_run_id)
        iteration_to_percent[i] = total_percent

    pprint(iteration_to_percent)

    starting = 1000
    for i in range(len(iteration_to_percent)):
        iteration = len(iteration_to_percent) - i - 1
        starting += starting * (iteration_to_percent[iteration] / 100)
    total_percent = starting / 1000 * 100 - 100
    print(f"Total profit percent is {total_percent} for {number_of_tests} iteration")


# x = """ABT,21
# BMY,20
# CVS,20
# DIS,20
# C,19
# DUK,19
# BAC,18
# CIX,18
# GM,18"""
# s = [i.split(',')[0] for i in x.split('\n') if i]


def main():
    calculate_predictor_score(
        SYMBOLS,
        PredictorOption.prophet,
        number_of_tests=15,
        style=InvestmentStyle(
            end_date=date(year=2024, month=7, day=10), hold_days=30, days_delta=13
        ),
        max_number_of_stocks_to_buy=3,
    )


if __name__ == "__main__":
    init_logger()
    main()
