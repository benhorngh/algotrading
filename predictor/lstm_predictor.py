import logging

import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler

import keras
from keras import layers


def predict(stock: pd.DataFrame, days: int):
    symbol = stock.columns[0]
    df = stock[symbol].values.reshape(-1, 1)
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaled_data = scaler.fit_transform(df)

    training_data_len = int(len(df) * 0.8)
    train_data = scaled_data[0:training_data_len, :]
    x_train, y_train = [], []

    for i in range(60, len(train_data)):
        x_train.append(train_data[i - 60 : i, 0])
        y_train.append(train_data[i, 0])

    x_train, y_train = np.array(x_train), np.array(y_train)
    x_train = np.reshape(x_train, (x_train.shape[0], x_train.shape[1], 1))

    input_shape = (x_train.shape[1], 1)
    model = keras.Sequential(
        [
            keras.Input(shape=input_shape),
            layers.LSTM(units=50, return_sequences=True),
            layers.LSTM(units=50, return_sequences=False),
            layers.Dense(units=25),
            layers.Dense(units=1),
        ]
    )
    model.compile(optimizer="adam", loss="mean_squared_error")
    model.fit(x_train, y_train, batch_size=1, epochs=1, verbose=0)

    test_data = scaled_data[training_data_len - 60 :, :]
    x_test = []
    y_test = df[training_data_len:]

    for i in range(60, len(test_data)):
        x_test.append(test_data[i - 60 : i, 0])

    x_test = np.array(x_test)
    x_test = np.reshape(x_test, (x_test.shape[0], x_test.shape[1], 1))

    predictions = model.predict(x_test)
    predictions = scaler.inverse_transform(predictions)

    rmse = np.sqrt(np.mean((predictions - y_test) ** 2))

    last_60_days = scaled_data[-60:]
    x_forecast = np.array(last_60_days)
    x_forecast = np.reshape(x_forecast, (x_forecast.shape[0], x_forecast.shape[1], 1))

    forecasted_prices = []

    for _ in range(days):
        price = model.predict(x_forecast)
        forecasted_prices.append(price[0][0])
        x_forecast = np.roll(x_forecast, -1)
        x_forecast[0][-1] = price

    forecasted_prices = scaler.inverse_transform(
        np.array(forecasted_prices).reshape(-1, 1)
    )
    logging.info(f"rmse is {rmse}")
    forecasted_prices = [float(p) for p in forecasted_prices]
    # forecasted_prices = [p - (rmse / 2) for p in forecasted_prices]
    return forecasted_prices
