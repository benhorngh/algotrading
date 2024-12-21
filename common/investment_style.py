from datetime import date

from pydantic import BaseModel


class InvestmentStyle(BaseModel):
    end_date: date
    hold_days: int
    days_delta: int  # days between investments


WEEKLY = InvestmentStyle(
    end_date=date(year=2024, month=7, day=24), hold_days=5, days_delta=7
)
BIWEEKLY = InvestmentStyle(
    end_date=date(year=2024, month=7, day=24), hold_days=10, days_delta=14
)
MONTHLY = InvestmentStyle(
    end_date=date(year=2024, month=7, day=9), hold_days=20, days_delta=28
)
BI_MONTHLY = InvestmentStyle(
    end_date=date(year=2024, month=7, day=1), hold_days=40, days_delta=56
)
CUSTOM = InvestmentStyle(
    end_date=date(year=2024, month=10, day=1), hold_days=30, days_delta=30
)
