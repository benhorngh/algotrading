import numpy as np
import pandas as pd

from sklearn.linear_model import LinearRegression


def predict(stock: pd.DataFrame, days: int) -> list[float]:
    prices = list(stock[stock.columns[0]].values)
    x = np.arange(len(prices)).reshape(-1, 1)
    y = stock
    model = LinearRegression()
    model.fit(x, y)

    future_x = np.arange(len(prices), len(prices) + days).reshape(-1, 1)
    forecast = model.predict(future_x)
    return forecast
