import random

import pandas as pd

from predictor import predictor_utils


def predict(stock: pd.DataFrame, hold_days: int) -> list[float]:
    current_value = predictor_utils.get_last_price(stock)
    future = [current_value for i in range(hold_days)]
    future[-1] = random.choice([current_value / 2, current_value * 2])
    return future
