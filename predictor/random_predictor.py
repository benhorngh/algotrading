import random

import pandas as pd

from common import utils


def predict(stock: pd.DataFrame, days: int) -> list[float]:
    max_day = utils.get_stock_last_day(stock)
    current_value = stock.loc[str(max_day)].values[0]
    future = [current_value for i in range(days)]
    future[-1] = random.choice([current_value / 2, current_value * 2])
    return future
