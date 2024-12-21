import pandas as pd
from prophet import Prophet

from common import stocks_data
from predictor import predictor_utils


# def predict(stock: pd.DataFrame, hold_days: int) -> list[float]:
#     symbol = predictor_utils.get_symbol(stock)
#     future_trading_dates = predictor_utils.get_future_trading_days(stock, hold_days)
#     future_trading_dates = [str(d) for d in future_trading_dates]
#     stock = stock.reset_index()
#     stock = stock.rename(columns={stocks_data.DATE_COL: "ds", symbol: "y"})
#     stock = stock.tail(450).copy()
#
#     # model = Prophet()
#     model = Prophet(daily_seasonality=True)
#
#     model.fit(stock)
#
#     future = pd.DataFrame(future_trading_dates, columns=["ds"])
#     forecast = model.predict(future)
#     # yhat_lower, yhat_upper, yhat
#     return forecast["yhat"].to_list()

def predict(stock: pd.DataFrame, hold_days: int) -> list[float]:
    validation_days = 10
    symbol = predictor_utils.get_symbol(stock)
    data = stock.head(len(stock) - validation_days)
    validation = stock.tail(validation_days)

    future_trading_dates = predictor_utils.get_future_trading_days(data, hold_days + validation_days)
    future_trading_dates = [str(d) for d in future_trading_dates]
    data = data.reset_index()
    validation = validation.reset_index()
    data = data.rename(columns={stocks_data.DATE_COL: "ds", symbol: "y"})
    validation = validation.rename(columns={stocks_data.DATE_COL: "ds", symbol: "y"})

    # model = Prophet()
    model = Prophet(daily_seasonality=True)

    model.fit(data)

    future = pd.DataFrame(future_trading_dates, columns=["ds"])
    forecast = model.predict(future)
    # yhat_lower, yhat_upper, yhat
    results = forecast["yhat"]
    expected = validation['y'][validation_days - 1]
    predicted = results[validation_days - 1]
    diff = predicted - expected
    percent_diff = diff * 100 / expected
    # print('percent_diff', symbol, percent_diff)
    if 3 > percent_diff > -3:
        return results.tail(hold_days).to_list()
    else:
        return [-1] * hold_days
