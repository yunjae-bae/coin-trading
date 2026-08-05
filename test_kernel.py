# -*- coding: utf-8 -*-
from __future__ import annotations

import os
import sys

import numpy as np
import pandas as pd

from coin_config import Config
from coin_kernel import (compute_signals, decide_last_bar, qqe, resample_ohlcv,
                         stop_loss_hit, warmup_bars, _rsi)

DATA = os.path.join(os.path.dirname(os.path.abspath(__file__)), "coin_hist_1h", "KRW-BTC.parquet")


def load() -> pd.DataFrame:
    df = pd.read_parquet(DATA)
    df.index = pd.to_datetime(df.index, utc=True)
    return df.sort_index()


def cfgs():
    return [Config(strategy="qqe"),
            Config(strategy="qqe_mod", rsi_period=6, factor=3.0)]


def test_converges_on_live_timeframe(df):
    cfg = Config(strategy="qqe_mod", rsi_period=6, factor=3.0, timeframe="6h")
    bars = resample_ohlcv(df, cfg.timeframe)
    need = warmup_bars(cfg)
    full = compute_signals(bars, cfg)
    mismatch = 0
    for end in range(len(bars) - 200, len(bars)):
        part = compute_signals(bars.iloc[max(0, end - need + 1):end + 1], cfg)
        ts = bars.index[end]
        if (bool(part.iloc[-1]["entry"]) != bool(full.loc[ts, "entry"]) or
                bool(part.iloc[-1]["exit"]) != bool(full.loc[ts, "exit"])):
            mismatch += 1
    assert mismatch == 0, f"6h: {need} bars insufficient; mismatches among 200 samples: {mismatch} mismatches"
    print(f"  [OK] Live timeframe (6h): last {need} bars match full history (200 samples from {len(bars)} bars)")


def test_no_lookahead(df):
    for cfg in cfgs():
        full = compute_signals(df, cfg)
        cut = compute_signals(df.iloc[:-200], cfg)
        common = cut.index
        for col in ("entry", "exit"):
            assert (full.loc[common, col].to_numpy() == cut[col].to_numpy()).all(), \
                f"{cfg.strategy}: Lookahead detected ({col})"
    print("  [OK] No lookahead")


def test_partial_data_converges(df):
    for cfg in cfgs():
        need = warmup_bars(cfg)
        full = compute_signals(df, cfg)
        mismatch = 0
        checked = 0
        for end in range(len(df) - 300, len(df)):
            part = compute_signals(df.iloc[max(0, end - need + 1):end + 1], cfg)
            ts = df.index[end]
            checked += 1
            if bool(part.iloc[-1]["entry"]) != bool(full.loc[ts, "entry"]) or \
               bool(part.iloc[-1]["exit"]) != bool(full.loc[ts, "exit"]):
                mismatch += 1
        assert mismatch == 0, \
            f"{cfg.strategy}: {need} bars insufficient: {checked} samples, {mismatch} signal mismatches"
        # Report the half-window margin without making it a pass criterion.
        half = need // 2
        half_mis = sum(
            1 for end in range(len(df) - 100, len(df))
            if (lambda p, t: bool(p.iloc[-1]["entry"]) != bool(full.loc[t, "entry"])
                or bool(p.iloc[-1]["exit"]) != bool(full.loc[t, "exit"]))(
                    compute_signals(df.iloc[max(0, end - half + 1):end + 1], cfg), df.index[end]))
        margin = "At least 2x history margin" if half_mis == 0 else f"Half window ({half} bars): {half_mis}/100 mismatches"
        print(f"  [OK] {cfg.strategy}: last {need} bars match full history ({checked} samples) — {margin}")


def test_rsi_reference(df):
    close = df["close"].astype(float).iloc[-2000:]
    n = 14
    delta = close.diff()
    gain = delta.clip(lower=0).to_numpy()
    loss = (-delta).clip(lower=0).to_numpy()
    ag = np.full(len(gain), np.nan)
    al = np.full(len(loss), np.nan)
    ag[n] = np.mean(gain[1:n + 1])
    al[n] = np.mean(loss[1:n + 1])
    for i in range(n + 1, len(gain)):
        ag[i] = (ag[i - 1] * (n - 1) + gain[i]) / n
        al[i] = (al[i - 1] * (n - 1) + loss[i]) / n
    ref = 100.0 - 100.0 / (1.0 + ag / al)

    ours = _rsi(close, n).to_numpy()
    both = ~np.isnan(ref) & ~np.isnan(ours)
    assert both.sum() > 1500, "Comparison window too short"
    diff = np.abs(ref[both] - ours[both]).max()
    assert diff < 1e-9, f"RSI mismatch; maximum error {diff}"
    assert (ours[both] >= 0).all() and (ours[both] <= 100).all(), "RSI out of range"
    print(f"  [OK] RSI matches Wilder definition (maximum error {diff:.2e})")


def test_live_matches_backtest(df):
    for cfg in cfgs():
        sig = compute_signals(df, cfg)
        for end in range(len(df) - 100, len(df)):
            sub = df.iloc[:end + 1]
            row = sig.iloc[end]
            act_flat, _ = decide_last_bar(sub, cfg, currently_long=False)
            act_long, _ = decide_last_bar(sub, cfg, currently_long=True, entry_price=0.0)
            assert (act_flat == "BUY") == bool(row["entry"]), f"{cfg.strategy}: Flat-position decision mismatch"
            assert (act_long == "SELL") == bool(row["exit"]), f"{cfg.strategy}: Long-position decision mismatch"
    print("  [OK] Backtest and live decisions agree")


def test_resample_drops_incomplete(df):
    full4 = resample_ohlcv(df, "4h")
    assert (full4.index.hour % 4 == 0).all(), "Four-hour bar boundaries are misaligned"
    last_start = full4.index[-1]
    trimmed = df[df.index < last_start + pd.Timedelta(hours=3)]
    part4 = resample_ohlcv(trimmed, "4h")
    assert last_start not in part4.index, "Incomplete four-hour bar retained"
    chunk = df[(df.index >= last_start) & (df.index < last_start + pd.Timedelta(hours=4))]
    assert full4.loc[last_start, "open"] == chunk["open"].iloc[0]
    assert full4.loc[last_start, "close"] == chunk["close"].iloc[-1]
    assert full4.loc[last_start, "high"] == chunk["high"].max()
    print(f"  [OK] Four-hour resampling (complete bars: {len(full4)}; incomplete bars removed)")


def test_stop_loss():
    assert stop_loss_hit(100.0, 95.0, 5.0) is True
    assert stop_loss_hit(100.0, 95.1, 5.0) is False
    assert stop_loss_hit(100.0, 1.0, 0.0) is False
    assert stop_loss_hit(0.0, 50.0, 5.0) is False
    print("  [OK] Stop-loss decisions")


def test_qqe_shape(df):
    q = qqe(df["close"].astype(float), 14, 5, 4.238)
    assert set(q.columns) == {"rsi_ma", "fast_tl", "trend"}
    assert q["trend"].dropna().isin([-1, 1]).all(), "Trend must be +1 or -1"
    assert q["fast_tl"].isna().sum() > 0, "Warmup values were not marked NaN"
    valid = q.dropna()
    assert len(valid) > len(q) - 200, "Valid signal window too short"
    print(f"  [OK] QQE output (valid {len(valid)}/{len(q)} bars)")


def main() -> int:
    if not os.path.exists(DATA):
        print(f"No data: {DATA} — Run upbit_candle_fetcher.py first.")
        return 1
    df = load()
    print(f"Data: {len(df)} bars {df.index.min()} ~ {df.index.max()}\n")
    test_qqe_shape(df)
    test_rsi_reference(df)
    test_stop_loss()
    test_resample_drops_incomplete(df)
    test_converges_on_live_timeframe(df)
    test_no_lookahead(df)
    test_live_matches_backtest(df)
    test_partial_data_converges(df)
    print("\nAll checks passed")
    return 0


if __name__ == "__main__":
    sys.exit(main())
