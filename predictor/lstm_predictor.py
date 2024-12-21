import os
from functools import lru_cache

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.optimizers.legacy import Adam
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
import pandas_market_calendars as mcal

from common import stocks_data
from predictor import predictor_utils

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'


def predict(stock: pd.DataFrame, hold_days: int) -> list[float]:
    symbol = predictor_utils.get_symbol(stock)
    stock = stock.reset_index()
    stock = stock.rename(columns={stocks_data.DATE_COL: "date", symbol: "price"})
    predictions = train_and_predict_stock_prices(stock)
    return predictions[:hold_days]


def calculate_rsi(prices, periods=14):
    """
    Calculate Relative Strength Index (RSI)

    Args:
        prices (pd.Series): Price series
        periods (int): Number of periods for RSI calculation

    Returns:
        pd.Series: RSI values
    """
    delta = prices.diff()

    up = delta.clip(lower=0)
    down = -1 * delta.clip(upper=0)

    roll_up = up.ewm(com=periods - 1).mean()
    roll_down = down.ewm(com=periods - 1).mean()

    rs = roll_up / roll_down
    rsi = 100.0 - (100.0 / (1.0 + rs))

    return rsi


def add_technical_indicators(df):
    """
    Add technical indicators to the DataFrame

    Args:
        df (pd.DataFrame): Input DataFrame with 'price' column

    Returns:
        pd.DataFrame: DataFrame with added technical indicators
    """
    df = df.copy()
    df['MA7'] = df['price'].rolling(window=7).mean()
    df['MA30'] = df['price'].rolling(window=30).mean()
    df['RSI'] = calculate_rsi(df['price'])

    # Calculate price differences (for stationarity)
    df['price_diff'] = df['price'].diff()

    return df.dropna()


def prepare_stock_data(df, look_back=60):
    """
    Prepare stock data for LSTM model

    Args:
        df (pd.DataFrame): DataFrame with technical indicators
        look_back (int): Number of previous time steps to use

    Returns:
        Tuple of prepared data and scaler
    """
    # Select features for prediction
    features = ['price_diff', 'MA7', 'MA30', 'RSI']
    X_data = df[features].values

    # Normalize the data
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaled_data = scaler.fit_transform(X_data)

    # Create sequences
    X, y = [], []
    for i in range(len(scaled_data) - look_back):
        X.append(scaled_data[i:i + look_back])
        y.append(scaled_data[i + look_back, 0])  # Predict price difference

    X = np.array(X)
    y = np.array(y)

    return X, y, scaler


def create_lstm_model(input_shape):
    """
    Create LSTM model architecture

    Args:
        input_shape (tuple): Shape of input data

    Returns:
        Compiled Keras model
    """
    model = Sequential([
        LSTM(100, return_sequences=True, input_shape=input_shape),
        Dropout(0.2),
        LSTM(50, return_sequences=False),
        Dropout(0.2),
        Dense(50, activation='relu'),
        Dense(1)
    ])

    model.compile(
        optimizer=Adam(learning_rate=0.001),
        loss='mean_squared_error',
        metrics=['mae']
    )
    return model


def train_and_predict_stock_prices(df, look_back=60, num_predict_days=30):
    """
    Train LSTM model and predict future stock prices

    Args:
        df (pd.DataFrame): Input DataFrame with 'date' and 'price' columns
        look_back (int): Number of previous time steps to use
        num_predict_days (int): Number of days to predict

    Returns:
        Tuple of predictions and prediction dates
    """
    # Convert date strings to datetime
    df['date'] = pd.to_datetime(df['date'])

    # Sort and add technical indicators
    df = df.sort_values('date').reset_index(drop=True)
    df = add_technical_indicators(df)

    # Prepare data
    X, y, scaler = prepare_stock_data(df, look_back)

    # Split data
    X_train, y_train = X, y

    # Create and train model
    model = create_lstm_model((X.shape[1], X.shape[2]))

    # Early stopping and learning rate reduction
    early_stopping = EarlyStopping(
        monitor='loss',
        patience=10,
        restore_best_weights=True
    )

    reduce_lr = ReduceLROnPlateau(
        monitor='loss',
        factor=0.2,
        patience=5,
        min_lr=0.00001
    )

    # Train model
    model.fit(
        X_train, y_train,
        epochs=50,
        batch_size=32,
        callbacks=[early_stopping, reduce_lr],
        verbose=0
    )

    # Prepare last sequence for prediction
    last_sequence = scaler.transform(
        df[['price_diff', 'MA7', 'MA30', 'RSI']].values[-look_back:]
    )
    last_sequence = last_sequence.reshape(1, look_back, 4)

    # Predict next days
    predictions = []
    current_seq = last_sequence.copy()

    for _ in range(num_predict_days):
        # Predict next value
        next_pred = model.predict(current_seq, verbose=0)
        predictions.append(next_pred[0, 0])

        # Update sequence
        current_seq = np.roll(current_seq, -1, axis=1)
        current_seq[0, -1, 0] = next_pred[0, 0]

    # Inverse transform predictions
    predicted_diffs = scaler.inverse_transform(
        np.column_stack([
            predictions,
            np.zeros((len(predictions), 3))  # Dummy values for other features
        ])
    )[:, 0]

    # Reconstruct prices
    last_known_price = df['price'].iloc[-1]
    predicted_prices = last_known_price + np.cumsum(predicted_diffs)

    return predicted_prices
