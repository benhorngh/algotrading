import pandas as pd
from prophet import Prophet

from common import stocks_data
from predictor import predictor_utils


def predict(stock: pd.DataFrame, hold_days: int) -> list[float]:
    stock = stock.tail(365)
    symbol = predictor_utils.get_symbol(stock)
    future_trading_dates = predictor_utils.get_future_trading_days(stock, hold_days)
    future_trading_dates = [str(d) for d in future_trading_dates]
    stock = stock.reset_index()
    stock = stock.rename(columns={stocks_data.DATE_COL: "ds", symbol: "y"})

    model = Prophet()

    model.fit(stock)

    future = pd.DataFrame(future_trading_dates, columns=["ds"])
    forecast = model.predict(future)

    return forecast["yhat"].to_list()
