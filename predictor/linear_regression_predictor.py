import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression

from predictor import predictor_utils


def predict(stock: pd.DataFrame, hold_days: int) -> list[float]:
    prices = predictor_utils.get_prices(stock)
    x = np.arange(len(prices)).reshape(-1, 1)
    y = stock
    model = LinearRegression()
    model.fit(x, y)

    future_x = np.arange(len(prices), len(prices) + hold_days).reshape(-1, 1)
    forecast = model.predict(future_x)
    return forecast
