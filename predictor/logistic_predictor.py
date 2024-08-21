import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

from predictor import predictor_utils

CHUNK_SIZE = 28


def get_x_y(
    stock: pd.DataFrame, hold_days: int
) -> (list[float], list[int], list[float]):
    symbol = predictor_utils.get_symbol(stock)
    stock = stock.reset_index()
    stock = stock.tail(len(stock) - len(stock) % CHUNK_SIZE)
    chunks = [stock[i : i + CHUNK_SIZE] for i in range(0, len(stock), CHUNK_SIZE)]
    chunks = [chunk.reset_index(drop=True) for chunk in chunks]

    y = []
    for i in range(len(chunks) - 1):
        next_chunk = chunks[i + 1]
        buy_price = next_chunk.loc[0].values[1]
        sell_price = next_chunk.loc[hold_days].values[1]
        if sell_price > buy_price:
            y.append(1)
        else:
            y.append(0)
    x = [c[symbol].to_list() for c in chunks[:-1]]
    x_future = chunks[-1][symbol].to_list()
    return x, y, x_future


def predict(stock: pd.DataFrame, hold_days: int) -> list[float]:
    current_value = predictor_utils.get_last_price(stock)

    x, y, x_future = get_x_y(stock, hold_days)

    model = LogisticRegression()
    scaler = StandardScaler()

    x = scaler.fit_transform(x)
    model.fit(x, y)

    x_future = [x_future]
    x_future = scaler.fit_transform(x_future)
    predictions = model.predict_proba(x_future)
    should_invest = predictions[0][1] > 0.6
    print(f"proba: {predictions[0][1]}")
    future = [current_value for i in range(hold_days)]
    if should_invest:
        future[-1] = current_value + (current_value * predictions[0][1])
    else:
        future[-1] = current_value / 2
    return future
