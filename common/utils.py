from datetime import date, timedelta
from functools import lru_cache

import pandas as pd
import pandas_market_calendars as mcal

US_STOCK_MARKET = "XNYS"


@lru_cache
def get_future_trading_dates(start_date: date, days: int) -> list[date]:
    start_date = start_date + timedelta(days=1)
    end_date = start_date + timedelta(days=days * 3)
    nyse = mcal.get_calendar(US_STOCK_MARKET)
    schedule = nyse.schedule(
        start_date,
        end_date,
    )
    trading_days = schedule.index.date.tolist()
    return trading_days[:days]


def get_stock_last_day(stocks: pd.DataFrame) -> date:
    return date.fromisoformat(max(stocks.index))


def create_simple_df(
    key_to_value: dict[str, float],
    key_name: str,
    value_name: str,
    sort_by_value: bool = True,
):
    data = list(key_to_value.items())
    if sort_by_value:
        data = sorted(data, key=lambda item: item[1], reverse=True)
    symbols = [d[0] for d in data]
    percent = [d[1] for d in data]
    df = pd.DataFrame(data={key_name: symbols, value_name: percent})
    return df


if __name__ == "__main__":
    future = get_future_trading_dates(date(year=2024, month=3, day=1), 7)
    print(future)
