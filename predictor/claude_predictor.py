import anthropic
import pandas as pd

from common.stocks_data import DATE_COL
from predictor import predictor_utils

KEY = "xxx"


client = anthropic.Anthropic(api_key=KEY)


def predict(stock: pd.DataFrame, hold_days: int) -> list[float]:
    symbol = predictor_utils.get_symbol(stock)
    stock = stock.tail(30)
    stock = stock.reset_index()
    prompt = f"Given the following sequence:\n\n"
    for _, row in stock.iterrows():
        prompt += f"{row[DATE_COL]}: ${row[symbol]:.2f}\n"
    prompt += f"\nPredict the next {hold_days} numbers. Provide only the predicted numbers as decimal numbers, separated by commas. No textual response, numbers and commas only."

    message = client.messages.create(
        model="claude-3-5-sonnet-20240620",
        max_tokens=1000,
        temperature=0,
        system="You are a theoretical analyst AI trained to predict number sequences with 100% accuracy.",
        messages=[{"role": "user", "content": [{"type": "text", "text": prompt}]}],
    )
    predicted_prices = [
        float(price.strip()) for price in message.content[0].text.split(",")
    ]
    return predicted_prices
