from __future__ import annotations

import sys
from dataclasses import replace

import numpy as np
import pandas as pd

from backtest_coin import REASON_BUY, simulate
from coin_config import Config
from coin_kernel import REASON_EXIT, REASON_STOP, warmup_bars

N = 610          # Exceed the 600-bar warmup floor so the final ten bars are simulated.
BASE = 100.0
WARM = 600


def _cfg(**kw) -> Config:
    flat = Config(position_pct=100.0, fee_rate=0.0, slippage_pct=0.0,
                  stop_loss_pct=0.0, cooldown_bars=0, min_order_krw=5000.0)
    return replace(flat, **kw)


def _frames(entries: list[int], exits: list[int], *, opens: dict[int, float] | None = None,
            lows: dict[int, float] | None = None) -> tuple[pd.DataFrame, pd.DataFrame]:
    idx = pd.date_range("2026-01-01", periods=N, freq="1h", tz="UTC")
    o = np.full(N, BASE)
    low = np.full(N, BASE)
    for i, v in (opens or {}).items():
        o[i] = v
    for i, v in (lows or {}).items():
        low[i] = v
    bars = pd.DataFrame({"open": o, "high": np.full(N, BASE), "low": low,
                         "close": np.full(N, BASE), "volume": np.ones(N)}, index=idx)
    entry = np.zeros(N, dtype=bool)
    entry[entries] = True
    exit_ = np.zeros(N, dtype=bool)
    exit_[exits] = True
    sig = pd.DataFrame({"close": np.full(N, BASE), "entry": entry, "exit": exit_}, index=idx)
    return bars, sig


def test_warm_window():
    assert min(warmup_bars(_cfg()), N - 2) == WARM, "Warmup changed; fixture bar indexes are invalid"
    print("  [OK] Simulation window assumptions")


def test_fills_at_next_bar_open():
    bars, sig = _frames(entries=[WARM], exits=[WARM + 3], opens={WARM + 1: 120.0, WARM + 4: 130.0})
    r = simulate(bars, sig, _cfg(), 1_000_000.0)
    assert len(r.trades) == 2, f"Expected one buy and one sell: {r.trades}"
    buy, sell = r.trades
    assert buy.price == 120.0, f"Expected next-open fill at 120, not signal close at 100: {buy.price}"
    assert buy.ts == bars.index[WARM + 1], "Fill timestamp is on the signal bar: lookahead bias"
    assert sell.price == 130.0 and sell.reason == REASON_EXIT
    assert buy.reason == REASON_BUY
    print("  [OK] Next-open fills without lookahead")


def test_fee_and_slippage():
    bars, sig = _frames(entries=[WARM], exits=[WARM + 3])
    r = simulate(bars, sig, _cfg(fee_rate=0.001, slippage_pct=1.0), 1_000_000.0)
    buy, sell = r.trades
    assert abs(buy.price - BASE * 1.01) < 1e-9, f"Buy price must include adverse slippage: {buy.price}"
    assert abs(sell.price - BASE * 0.99) < 1e-9, f"Sell price must include adverse slippage: {sell.price}"
    assert abs(buy.krw_after) < 1e-6, f"Cash remains despite 100% allocation: {buy.krw_after}"
    assert abs(r.fees - (buy.fee_krw + sell.fee_krw)) < 1e-9, "Total fees differ from the sum of fill fees"
    assert r.round_trips == 1 and r.win_rate == 0.0, "A trade losing to fees and slippage was counted as a win"
    print("  [OK] Fees and adverse slippage on both sides")


def test_stop_inside_bar():
    bars, sig = _frames(entries=[WARM], exits=[], lows={WARM + 2: 89.0})
    r = simulate(bars, sig, _cfg(stop_loss_pct=10.0), 1_000_000.0)
    stop = r.trades[1]
    assert stop.reason == REASON_STOP and r.stops == 1
    assert stop.price == 90.0, f"Expected fill at stop level 90: {stop.price}"
    assert stop.ts == bars.index[WARM + 2], "Stop delayed to the next bar; expected intrabar fill"
    print("  [OK] Intrabar stop without waiting for close")


def test_stop_gap_fills_at_open():
    bars, sig = _frames(entries=[WARM], exits=[], opens={WARM + 2: 85.0}, lows={WARM + 2: 80.0})
    r = simulate(bars, sig, _cfg(stop_loss_pct=10.0), 1_000_000.0)
    assert r.trades[1].price == 85.0, f"Expected gap-open fill at 85, not stop level 90: {r.trades[1].price}"
    print("  [OK] Gap-down stop filled at open")


def test_cooldown_blocks_reentry():
    bars, sig = _frames(entries=[WARM, WARM + 2, WARM + 5], exits=[WARM + 1])
    held = simulate(bars, sig, _cfg(cooldown_bars=2), 1_000_000.0)
    free = simulate(bars, sig, _cfg(cooldown_bars=0), 1_000_000.0)
    assert len(held.trades) == 3 and len(free.trades) == 3

    assert held.trades[2].ts == bars.index[WARM + 6], \
        f"Reentered after one bar despite a two-bar cooldown: {held.trades[2].ts}"
    assert free.trades[2].ts == bars.index[WARM + 3], \
        f"Entry delayed despite zero cooldown: {free.trades[2].ts}"
    print("  [OK] Cooldown blocks reentry")


def test_min_order_krw_skips_buy():
    bars, sig = _frames(entries=[WARM], exits=[])
    r = simulate(bars, sig, _cfg(min_order_krw=5000.0), 1_000.0)
    assert r.trades == [], "Bought below the minimum order amount"
    assert r.final == 1_000.0 and r.round_trips == 0
    print("  [OK] Skip buys below minimum order amount")


def test_index_mismatch_raises():
    bars, sig = _frames(entries=[WARM], exits=[])
    try:
        simulate(bars, sig.iloc[:-1], _cfg(), 1_000_000.0)
    except ValueError:
        print("  [OK] Reject mismatched bar and signal indexes")
        return
    raise AssertionError("Mismatched indexes were accepted")


def main() -> int:
    test_warm_window()
    test_fills_at_next_bar_open()
    test_fee_and_slippage()
    test_stop_inside_bar()
    test_stop_gap_fills_at_open()
    test_cooldown_blocks_reentry()
    test_min_order_krw_skips_buy()
    test_index_mismatch_raises()
    print("\nAll checks passed")
    return 0


if __name__ == "__main__":
    sys.exit(main())
