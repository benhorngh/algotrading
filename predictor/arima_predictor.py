import pandas as pd
import numpy as np
from pmdarima import auto_arima
from datetime import datetime

from common import stocks_data
from predictor import predictor_utils


def predict(stock: pd.DataFrame, hold_days: int = 30) -> list[float]:
    symbol = predictor_utils.get_symbol(stock)
    stock = stock.reset_index()
    stock = stock.rename(columns={stocks_data.DATE_COL: "date", symbol: 'price'})

    # Convert date strings to datetime
    stock['date'] = pd.to_datetime(stock['date'])

    # Sort by date to ensure time series is in correct order
    stock = stock.sort_values('date')

    # Calculate returns instead of using raw prices
    stock['returns'] = stock['price'].pct_change()

    # Drop the first row with NaN return
    returns = stock['returns'].dropna()

    # Create the ARIMA model with adjusted parameters
    model = auto_arima(
        returns,
        start_p=1,
        max_p=5,
        start_q=1,
        max_q=5,
        m=1,
        start_P=0,
        seasonal=False,
        d=1,
        D=1,
        trace=False,
        # error_action='ignore',
        suppress_warnings=False,
        stepwise=True,
        random_state=42
    )

    # Predict future returns
    forecast_returns = model.predict(n_periods=hold_days)

    # Convert returns back to prices
    last_price = stock['price'].iloc[-1]
    prices = [last_price]

    for return_value in forecast_returns:
        next_price = prices[-1] * (1 + return_value)
        prices.append(next_price)

    # Remove the initial price and round to 2 decimal places
    predictions = [round(float(x), 2) for x in prices[1:]]

    return predictions
