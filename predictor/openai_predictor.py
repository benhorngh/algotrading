import pandas as pd

from openai import OpenAI

from common.stocks_data import DATE_COL

client = OpenAI()


def predict(stock: pd.DataFrame, days: int) -> list[float]:
    symbol = stock.columns[0]
    stock = stock.tail(30)
    stock = stock.reset_index()
    prompt = f"Given the following sequence:\n\n"
    for _, row in stock.iterrows():
        prompt += f"{row[DATE_COL]}: ${row[symbol]:.2f}\n"
    prompt += f"\nPredict the next {days} numbers. Provide only the predicted numbers as decimal numbers, separated by commas. No textual response, numbers and commas only."

    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {
                "role": "system",
                "content": "You are a financial analyst AI trained to predict stock prices based on historical data with 100% accuracy.",
            },
            {"role": "user", "content": prompt},
        ],
    )
    predicted_prices = [
        float(price.strip())
        for price in response.choices[0].message["content"].split(",")
    ]
    return predicted_prices
