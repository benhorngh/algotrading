from datetime import datetime

import pandas as pd
from sklearn.ensemble import RandomForestRegressor

from common.stocks_data import DATE_COL
from predictor import predictor_utils

FEATURES = ["day", "day_of_week"]


def _features(stock: pd.DataFrame) -> pd.DataFrame:
    stock["day_of_week"] = stock[DATE_COL].apply(lambda x: x.weekday())
    stock["pd_date"] = pd.to_datetime(stock[DATE_COL])
    stock["day"] = stock["pd_date"].dt.day
    return stock[FEATURES]


def predict(stock: pd.DataFrame, hold_days: int) -> list[float]:
    symbol = predictor_utils.get_symbol(stock)
    future_trading_dates = predictor_utils.get_future_trading_days(stock, hold_days)

    stock = stock.reset_index()
    y_train = stock[symbol]
    x_train = stock.copy()
    x_train[DATE_COL] = x_train[DATE_COL].apply(
        lambda x: datetime.fromisoformat(x).date()
    )
    x_train = _features(x_train)

    model = RandomForestRegressor()
    model.fit(x_train, y_train)

    x_future = pd.DataFrame(data={DATE_COL: future_trading_dates})
    x_future = _features(x_future)
    forecast = model.predict(x_future)
    return forecast
