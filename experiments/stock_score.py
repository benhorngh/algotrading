from datetime import timedelta
from pprint import pprint

from common.investment_style import InvestmentStyle
from steps import strategy_step, run_workflow
from steps.prediction_step import PredictorOption
from common import stocks, investment_style
from common import cache_manager, utils


def calculate_symbol_score(
    symbols: list[str],
    predictor_option: PredictorOption,
    number_of_tests: int,
    style: InvestmentStyle,
):
    end_date = style.end_date
    hold_days = style.hold_days
    max_number_of_stocks_to_buy = len(symbols)
    symbol_to_score = {s: 0 for s in symbols}
    for i in range(number_of_tests):
        predictor_run_id = run_workflow.run(
            cutoff_date=end_date,
            predictor_option=predictor_option,
            symbols=symbols,
            max_number_of_stocks_to_buy=max_number_of_stocks_to_buy,
            hold_days=hold_days,
        )
        ideal_run_id = run_workflow.run(
            cutoff_date=end_date,
            predictor_option=PredictorOption.ideal,
            symbols=symbols,
            max_number_of_stocks_to_buy=max_number_of_stocks_to_buy,
            hold_days=hold_days,
        )

        predictor_strategy = strategy_step.get_run_strategy(predictor_run_id)
        ideal_strategy = strategy_step.get_run_strategy(ideal_run_id)
        for symbol in symbols:
            if symbol in ideal_strategy and symbol in predictor_strategy:
                symbol_to_score[symbol] += 1
            elif symbol not in ideal_strategy and symbol not in predictor_strategy:
                symbol_to_score[symbol] += 1
        end_date = end_date - timedelta(days=style.days_delta)
    pprint(symbol_to_score)
    save_scores(symbol_to_score)


def save_scores(symbol_to_score: dict[str, float]):
    df = utils.create_simple_df(symbol_to_score, "symbol", "score")
    cache_manager.save_tmp_file(
        cache_manager.CacheFile.symbol_prediction_score, df, with_index=False
    )


def main():
    calculate_symbol_score(
        stocks.SYMBOLS[:10],
        PredictorOption.lstm,
        number_of_tests=10,
        style=investment_style.WEEKLY,
    )


if __name__ == "__main__":
    main()
