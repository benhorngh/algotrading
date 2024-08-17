from datetime import datetime

import pandas as pd
from prophet import Prophet

from common import utils, stocks_data


# def _add_features(stock: pd.DataFrame):
#     stock["date"] = stock["ds"].apply(lambda x: datetime.fromisoformat(x).date())
#     stock["day_of_week"] = stock["date"].apply(lambda x: x.weekday())
#     stock = pd.get_dummies(stock, columns=["day_of_week"], prefix="dow")
#     for i in range(7):
#         if f"dow_{i}" not in stock.columns:
#             stock[f"dow_{i}"] = 0
#     return stock


def predict(stock: pd.DataFrame, days: int) -> list[float]:
    # stock = stock.tail(365)
    max_day = utils.get_stock_last_day(stock)
    future_trading_dates = [
        str(d) for d in utils.get_future_trading_dates(max_day, days)
    ]
    column_name = stock.columns[0]
    stock = stock.reset_index()
    stock = stock.rename(columns={stocks_data.DATE_COL: "ds", column_name: "y"})
    # stock = _add_features(stock)

    model = Prophet()

    # for i in range(7):
    #     model.add_regressor(f"dow_{i}")

    model.fit(stock)

    future = pd.DataFrame(future_trading_dates, columns=["ds"])
    # future = _add_features(future)
    forecast = model.predict(future)

    return forecast["yhat"].to_list()
