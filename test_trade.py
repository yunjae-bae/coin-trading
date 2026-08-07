from __future__ import annotations

import csv
import os
import pathlib
import shutil
import sys
import tempfile
from dataclasses import replace
from datetime import datetime, timedelta

import numpy as np
import pandas as pd

import autotrade_coin
from autotrade_coin import Bot
from bot_state import Position, StateStore, StopFlag, now_kst
from coin_config import Config
from coin_kernel import resample_ohlcv
from fake_exchange import FakeExchange

BAR = "2026-08-04T16:00:00+00:00"


def _bot(tmp: str, api: FakeExchange, **cfg_kw) -> Bot:
    cfg = replace(Config(dry_run=False, access_key="k", secret_key="s",
                         position_pct=50.0, fee_rate=0.0005, min_order_krw=5000.0,
                         price_guard_pct=5.0, stop_loss_pct=5.0), **cfg_kw)
    bot = Bot(cfg, api)
    bot.store = StateStore(os.path.join(tmp, f"state-{id(api)}.json"))
    bot.alert = lambda text: None
    return bot


def test_position_pct_and_cap(tmp):
    api = FakeExchange(krw=1_000_000.0, fill_qty=5.0, fill_avg=100_000.0)
    _bot(tmp, api, position_pct=50.0).buy(1_000_000.0, 100_000.0, BAR, "entry", 100_000.0)
    spent = api.orders[0].amount
    assert 499_000 <= spent <= 500_000, f"Expected 50% allocation; spent {spent:,.0f} KRW"

    capped = FakeExchange(krw=1_000_000.0, fill_qty=5.0, fill_avg=100_000.0)
    _bot(tmp, capped, position_pct=100.0, max_order_krw=200_000.0).buy(
        1_000_000.0, 100_000.0, BAR, "entry", 100_000.0)
    assert capped.orders[0].amount <= 200_000.0, f"Exceeded maximum order amount: {capped.orders[0].amount}"
    print("  [OK] Allocation·Maximum buy amount")


def test_price_guard_holds_buy(tmp):
    api = FakeExchange(krw=1_000_000.0, fill_qty=5.0, fill_avg=100_000.0)
    bot = _bot(tmp, api, price_guard_pct=5.0)
    bot.buy(1_000_000.0, 110_000.0, BAR, "entry", 100_000.0)
    assert api.orders == [], "Entered beyond the price guard"

    bot.buy(1_000_000.0, 102_000.0, BAR, "entry", 100_000.0)
    assert len(api.orders) == 1, "Did not buy within the allowed range"
    print("  [OK] Price guard defers only excessive gaps")


def test_unconfirmed_fill_uses_balance_delta(tmp):
    autotrade_coin.BALANCE_SETTLE_SEC = 0.0
    api = FakeExchange(krw=1_000_000.0, fill_state="wait", fill_qty=0.0, fill_avg=0.0)
    bot = _bot(tmp, api)
    api.fill_qty = 0.0
    bot.buy(1_000_000.0, 100_000.0, BAR, "entry", 100_000.0)
    assert not bot.store.state.position.is_open, "Recorded a buy without a balance increase"

    moved = FakeExchange(krw=1_000_000.0, fill_state="wait", fill_qty=0.0, fill_avg=0.0)
    bot2 = _bot(tmp, moved)
    original = moved.buy_market

    def buy_and_settle(market: str, krw: float, identifier: str) -> str:
        out = original(market, krw, identifier)
        moved.coin += 4.9
        return out

    moved.buy_market = buy_and_settle
    bot2.buy(1_000_000.0, 100_000.0, BAR, "entry", 100_000.0)
    pos = bot2.store.state.position
    assert pos.is_open and abs(pos.qty - 4.9) < 1e-9, f"Failed to recognize the balance increase: {pos.qty}"
    print("  [OK] Recover unconfirmed fills from balance changes")


def test_dust_is_abandoned(tmp):
    api = FakeExchange(coin=0.00001, price=100_000.0)
    bot = _bot(tmp, api, min_order_krw=5000.0)
    bot.store.state.position = Position(qty=0.00001, entry_price=100_000.0)
    bot.store.state.pending_exit = True
    bot.sell(100_000.0, BAR, "exit")
    assert api.orders == [], "Submitted an order below the minimum amount"
    assert not bot.store.state.position.is_open and not bot.store.state.pending_exit, \
        "Dust was retained and would trigger repeated retries"
    print("  [OK] Abandon dust without endless retries")


def test_exchange_rejection_stops_retrying(tmp):
    api = FakeExchange(coin=1.0, price=100_000.0, reject_sell_code="under_min_total_ask")
    bot = _bot(tmp, api)
    bot.store.state.position = Position(qty=1.0, entry_price=100_000.0)
    bot.store.state.pending_exit = True
    bot.sell(100_000.0, BAR, "exit")
    assert not bot.store.state.position.is_open and not bot.store.state.pending_exit, \
        "Rejected exit remained pending and would retry indefinitely"
    print("  [OK] Stop retries after definitive sell rejection")


def test_partial_fill_keeps_pending_exit(tmp):
    api = FakeExchange(coin=1.0, price=100_000.0, fill_qty=0.7, fill_avg=100_000.0,
                       remain_after_sell=0.3)
    bot = _bot(tmp, api)
    bot.store.state.position = Position(qty=1.0, entry_price=100_000.0)
    bot.sell(100_000.0, BAR, "exit")
    st = bot.store.state
    assert st.pending_exit, "Partial fill did not retain pending exit"
    assert abs(st.position.qty - 0.3) < 1e-9, f"Incorrect residual quantity: {st.position.qty}"

    again = StateStore(bot.store.path)
    again.load()
    assert again.state.pending_exit and abs(again.state.position.qty - 0.3) < 1e-9
    print("  [OK] Partial-fill tracking survives restart")


def test_sells_only_bot_quantity(tmp):
    api = FakeExchange(coin=10.0, price=100_000.0, fill_qty=1.0, fill_avg=100_000.0,
                       remain_after_sell=9.0)
    bot = _bot(tmp, api)
    bot.store.state.position = Position(qty=1.0, entry_price=100_000.0)
    bot.sell(100_000.0, BAR, "exit")
    assert abs(api.orders[0].amount - 1.0) < 1e-8, \
        f"Expected bot-owned quantity 1.0, sold {api.orders[0].amount}; unrelated holdings were sold"
    print("  [OK] Sell only bot-owned quantity")


def _flat_bars(n: int = 4000) -> pd.DataFrame:
    idx = pd.date_range("2026-01-01", periods=n, freq="1h", tz="UTC")
    v = np.full(n, 100_000.0)
    return pd.DataFrame({"open": v, "high": v, "low": v, "close": v, "volume": np.ones(n)},
                        index=idx)


def _just_after_last_bar(bars_1h: pd.DataFrame) -> datetime:
    last = resample_ohlcv(bars_1h, "6h").index[-1].to_pydatetime()
    return last + timedelta(hours=6, minutes=1)


def test_same_bar_is_not_ordered_twice(tmp):
    bars = _flat_bars()
    api = FakeExchange(krw=1_000_000.0, price=100_000.0, bars=bars,
                       fill_qty=5.0, fill_avg=100_000.0)
    api.now = _just_after_last_bar(bars)
    bot = _bot(tmp, api, timeframe="6h", daily_loss_limit_pct=0.0)

    def always_buy(bars: pd.DataFrame) -> tuple[str | None, str]:
        return "BUY", "Test entry"                     # Test engine wiring independently of indicator calculations.

    bot._decide = always_buy

    bot.cycle()
    assert len(api.orders) == 1, f"No order on the first cycle: {api.orders}"
    bot.store.state.position = Position()
    bot.cycle()
    assert len(api.orders) == 1, "Duplicate order from processing the same bar twice"
    print("  [OK] Prevent duplicate bar processing")


def test_daily_loss_halts_bot(tmp):
    bars = _flat_bars()
    api = FakeExchange(krw=100_000.0, price=100_000.0, bars=bars)
    api.now = _just_after_last_bar(bars)
    bot = _bot(tmp, api, timeframe="6h", daily_loss_limit_pct=10.0)
    bot.stop = StopFlag(os.path.join(tmp, f"stop-{id(api)}.txt"))
    bot.store.state.day_date = now_kst().strftime("%Y-%m-%d")
    bot.store.state.day_start_equity = 1_000_000.0

    try:
        bot.cycle()
    except SystemExit:
        assert bot.stop.active, "Halted without writing a stop file"
        assert "limit" in bot.stop.reason()
        print("  [OK] Halt on daily loss limit")
        return
    raise AssertionError("Continued running after limit breach")


def test_watch_stops_out_between_bars(tmp):
    api = FakeExchange(coin=1.0, price=94_000.0, fill_qty=1.0, fill_avg=94_000.0)
    bot = _bot(tmp, api, stop_loss_pct=5.0)
    bot.store.state.position = Position(qty=1.0, entry_price=100_000.0)

    bot.watch_between_bars()
    assert len(api.orders) == 1, f"Did not sell after crossing the stop: {api.orders}"
    assert not bot.store.state.position.is_open

    calm = FakeExchange(coin=1.0, price=96_000.0)
    bot2 = _bot(tmp, calm, stop_loss_pct=5.0)
    bot2.store.state.position = Position(qty=1.0, entry_price=100_000.0)
    bot2.watch_between_bars()
    assert calm.orders == [], "Sold above the stop level"
    print("  [OK] Stop loss between bars without waiting for close")


def test_watch_finishes_pending_exit(tmp):
    api = FakeExchange(coin=0.3, price=100_000.0, fill_qty=0.3, fill_avg=100_000.0)
    bot = _bot(tmp, api, stop_loss_pct=0.0)          # Pending exits must finish even with stop losses disabled.
    bot.store.state.position = Position(qty=0.3, entry_price=100_000.0)
    bot.store.state.pending_exit = True

    bot.watch_between_bars()
    assert len(api.orders) == 1, "Pending exit did not close the remainder"
    assert not bot.store.state.pending_exit
    print("  [OK] Finish pending exits between bars")


def test_watch_identifier_differs_per_minute(tmp):
    bot = _bot(tmp, FakeExchange(), stop_loss_pct=5.0)
    a = bot._identifier("w202608050026", "ask")
    b = bot._identifier("w202608050027", "ask")
    bar = bot._identifier("2026-08-05T00:00:00+00:00", "ask")
    assert a != b and a != bar, f"Order identifiers collide: {a} / {b} / {bar}"
    print("  [OK] Distinct identifiers between bars")


def test_order_log_is_readable_in_excel(tmp):
    path = os.path.join(tmp, "isolated-orders.csv")
    autotrade_coin.ORDERS_PATH = path
    autotrade_coin.log_order({"timestamp": "2026-08-04 01:00:00", "side": "buy", "reason": "entry",
                              "market": "KRW-BTC", "quantity": "0.1", "fill_price": "100000",
                              "amount": "10000", "mode": "dry run", "order_id": "x"})
    autotrade_coin.log_order({"side": "sell", "unknown_column": "ignored_value"})

    raw = pathlib.Path(path).read_bytes()
    assert raw.startswith(b"\xef\xbb\xbf"), "Missing BOM for spreadsheet compatibility"
    with open(path, encoding="utf-8-sig", newline="") as f:
        rows = list(csv.reader(f))
    assert rows[0] == autotrade_coin.ORDER_HEADER, f"Unexpected column order: {rows[0]}"
    assert len(rows) == 3, "Duplicate header or missing row"
    assert rows[2][1] == "sell" and rows[2][0] == "", "Missing values should be empty cells"
    assert "ignored_value" not in "".join(rows[2]), "Unknown column leaked into output"
    print("  [OK] Order CSV encoding, column order, and single header")


def main() -> int:
    tmp = tempfile.mkdtemp(prefix="coinbot-trade-")
    autotrade_coin.OUT_DIR = tmp
    autotrade_coin.LOG_PATH = os.path.join(tmp, "\uae30\ub85d.log")
    autotrade_coin.ORDERS_PATH = os.path.join(tmp, "\uc8fc\ubb38\ub0b4\uc5ed.csv")
    try:
        test_position_pct_and_cap(tmp)
        test_price_guard_holds_buy(tmp)
        test_unconfirmed_fill_uses_balance_delta(tmp)
        test_dust_is_abandoned(tmp)
        test_exchange_rejection_stops_retrying(tmp)
        test_partial_fill_keeps_pending_exit(tmp)
        test_sells_only_bot_quantity(tmp)
        test_same_bar_is_not_ordered_twice(tmp)
        test_daily_loss_halts_bot(tmp)
        test_watch_stops_out_between_bars(tmp)
        test_watch_finishes_pending_exit(tmp)
        test_watch_identifier_differs_per_minute(tmp)
        test_order_log_is_readable_in_excel(tmp)
        print("\nAll checks passed")
        return 0
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


if __name__ == "__main__":
    sys.exit(main())
