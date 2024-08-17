import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

from common import utils

CHUNK_SIZE = 28


def get_x_y(stock: pd.DataFrame, days: int) -> (list[float], list[int], list[float]):
    symbol = stock.columns[0]
    stock = stock.reset_index()
    stock = stock.tail(len(stock) - len(stock) % CHUNK_SIZE)
    chunks = [stock[i : i + CHUNK_SIZE] for i in range(0, len(stock), CHUNK_SIZE)]
    chunks = [chunk.reset_index(drop=True) for chunk in chunks]

    y = []
    for i in range(len(chunks) - 1):
        next_chunk = chunks[i + 1]
        buy_price = next_chunk.loc[0].values[1]
        sell_price = next_chunk.loc[days].values[1]
        if sell_price > buy_price:
            y.append(1)
        else:
            y.append(0)
    x = [c[symbol].to_list() for c in chunks[:-1]]
    x_future = chunks[-1][symbol].to_list()
    return x, y, x_future


# class Logistic:
#     model = LogisticRegression(class_weight='balanced', max_iter=1000, solver='saga')
#     scaler = StandardScaler()
#     fitted = False


# def pre_predict(stocks: pd.DataFrame, days: int):
#     if Logistic.fitted:
#         return
#     x, y = [], []
#     for symbol in stocks.columns:
#         _x, _y, _x_future = get_x_y(stocks[[symbol]], days)
#         x += _x
#         y += _y
#
#     x = Logistic.scaler.fit_transform(x)
#     Logistic.model.fit(x, y)
#     Logistic.fitted = True


def predict(stock: pd.DataFrame, days: int) -> list[float]:
    max_day = utils.get_stock_last_day(stock)
    current_value = stock.loc[str(max_day)].values[0]

    x, y, x_future = get_x_y(stock, days)

    model = LogisticRegression()
    scaler = StandardScaler()

    x = scaler.fit_transform(x)
    model.fit(x, y)

    x_future = [x_future]
    x_future = scaler.fit_transform(x_future)
    predictions = model.predict_proba(x_future)
    should_invest = predictions[0][1] > 0.6
    print(f"proba: {predictions[0][1]}")
    future = [current_value for i in range(days)]
    if should_invest:
        future[-1] = current_value + (current_value * predictions[0][1])
    else:
        future[-1] = current_value / 2
    return future
