from datetime import date

from pydantic import BaseModel


class InvestmentStyle(BaseModel):
    end_date: date
    hold_days: int
    days_delta: int


WEEKLY = InvestmentStyle(
    end_date=date(year=2024, month=7, day=24), hold_days=5, days_delta=7
)
BIWEEKLY = InvestmentStyle(
    end_date=date(year=2024, month=7, day=24), hold_days=10, days_delta=14
)
MONTHLY = InvestmentStyle(
    end_date=date(year=2024, month=7, day=9), hold_days=20, days_delta=28
)
