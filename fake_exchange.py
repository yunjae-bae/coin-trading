from __future__ import annotations

from dataclasses import dataclass, field
from datetime import UTC, datetime

import pandas as pd

from upbit_api import UpbitError


@dataclass
class Order:
    side: str
    amount: float          # KRW for buys; coin quantity for sells.
    identifier: str


@dataclass
class FakeExchange:

    krw: float = 10_000_000.0
    coin: float = 0.0
    price: float | None = 100_000.0
    now: datetime = field(default_factory=lambda: datetime(2026, 8, 4, 16, 0, 5, tzinfo=UTC))
    bars: pd.DataFrame | None = None

    fill_state: str = "done"
    fill_qty: float = 0.0
    fill_avg: float = 0.0
    remain_after_sell: float | None = None      # None means a full fill.
    reject_sell_code: str = ""
    orders: list[Order] = field(default_factory=list)


    def now_utc(self) -> datetime:
        return self.now

    def candles_1h(self, market: str, count: int) -> pd.DataFrame:
        if self.bars is None:
            raise UpbitError(f"{market} has no configured market data")
        return self.bars.tail(count)

    def last_price(self, market: str) -> float | None:
        return self.price

    def balances(self, market: str) -> tuple[float, float]:
        return self.krw, self.coin


    def buy_market(self, market: str, krw: float, identifier: str) -> str:
        self.orders.append(Order("bid", krw, identifier))
        self.coin += self.fill_qty
        self.krw -= krw
        return f"uuid-{len(self.orders)}"

    def sell_market(self, market: str, qty: float, identifier: str) -> str:
        if self.reject_sell_code:
            raise UpbitError("Exchange rejected the sell order", "", self.reject_sell_code, 400)
        self.orders.append(Order("ask", qty, identifier))
        self.coin = self.remain_after_sell if self.remain_after_sell is not None else 0.0
        return f"uuid-{len(self.orders)}"

    def wait_done(self, identifier: str) -> tuple[str, float, float]:
        return self.fill_state, self.fill_qty, self.fill_avg
