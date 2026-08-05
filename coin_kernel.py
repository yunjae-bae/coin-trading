# -*- coding: utf-8 -*-
from __future__ import annotations

from typing import Optional, Tuple

import numpy as np
import pandas as pd

from coin_config import Config, tf_hours

REASON_ENTRY = "QQE bullish reversal"
REASON_EXIT = "QQE bearish reversal"
REASON_STOP = "Stop level reached"


def _seeded_ewm(s: pd.Series, alpha: float, seed_len: int) -> pd.Series:
    """Seed with the initial mean; earlier values are NaN."""
    v = s.to_numpy(dtype=float)
    out = np.full(len(v), np.nan)
    valid = np.flatnonzero(~np.isnan(v))
    if len(valid) < seed_len:
        return pd.Series(out, index=s.index)
    first = valid[0]
    start = first + seed_len - 1
    if start >= len(v):
        return pd.Series(out, index=s.index)
    out[start] = np.mean(v[first:start + 1])
    for i in range(start + 1, len(v)):
        out[i] = alpha * v[i] + (1.0 - alpha) * out[i - 1]
    return pd.Series(out, index=s.index)


def _rma(s: pd.Series, length: int) -> pd.Series:
    """Wilder smoothing matching Pine ta.rma."""
    return _seeded_ewm(s, 1.0 / length, length)


def _ema(s: pd.Series, length: int) -> pd.Series:
    """Exponential moving average matching Pine ta.ema."""
    return _seeded_ewm(s, 2.0 / (length + 1.0), length)


def _rsi(close: pd.Series, length: int) -> pd.Series:
    """Wilder RSI matching Pine ta.rsi."""
    delta = close.diff()
    up = _rma(delta.clip(lower=0.0), length)
    down = _rma((-delta).clip(lower=0.0), length)
    rs = up / down
    out = 100.0 - 100.0 / (1.0 + rs)
    out = out.where(down != 0.0, 100.0)
    out = out.where(up != 0.0, 0.0)
    return out.where(up.notna() & down.notna())


def _cross(a: pd.Series, b: pd.Series) -> pd.Series:
    """Direction-independent crossover matching Pine ta.cross."""
    pa, pb = a.shift(1), b.shift(1)
    return (((pa < pb) & (a >= b)) | ((pa > pb) & (a <= b))).fillna(False)


def qqe(close: pd.Series, rsi_period: int, sf: int, factor: float) -> pd.DataFrame:
    """Return (rsi_ma, fast_tl, trend); initial trend is +1."""
    wilders = rsi_period * 2 - 1
    rsi_ma = _ema(_rsi(close.astype(float), rsi_period), sf)
    atr_rsi = (rsi_ma.shift(1) - rsi_ma).abs()
    dar = _ema(_ema(atr_rsi, wilders), wilders) * factor

    idx = rsi_ma.to_numpy(dtype=float)
    d = dar.to_numpy(dtype=float)
    n = len(idx)
    longband = np.zeros(n)
    shortband = np.zeros(n)
    for i in range(1, n):
        if np.isnan(idx[i]) or np.isnan(d[i]):
            continue
        newlong = idx[i] - d[i]
        newshort = idx[i] + d[i]
        prev_idx = idx[i - 1]
        if prev_idx > longband[i - 1] and idx[i] > longband[i - 1]:
            longband[i] = max(longband[i - 1], newlong)
        else:
            longband[i] = newlong
        if prev_idx < shortband[i - 1] and idx[i] < shortband[i - 1]:
            shortband[i] = min(shortband[i - 1], newshort)
        else:
            shortband[i] = newshort

    lb = pd.Series(longband, index=close.index)
    sb = pd.Series(shortband, index=close.index)
    # Wait for valid bands: zero bands during NaN warmup could create false crosses.
    ready = dar.notna().to_numpy()
    cross_down = _cross(lb.shift(1), rsi_ma).to_numpy()
    cross_up = _cross(rsi_ma, sb.shift(1)).to_numpy()

    trend = np.ones(n, dtype=int)
    for i in range(1, n):
        if not ready[i]:
            continue
        if cross_up[i]:
            trend[i] = 1
        elif cross_down[i]:
            trend[i] = -1
        else:
            trend[i] = trend[i - 1]

    fast_tl = np.where(trend == 1, longband, shortband)
    out = pd.DataFrame({"rsi_ma": rsi_ma, "fast_tl": fast_tl, "trend": trend}, index=close.index)
    out.loc[~ready, ["fast_tl", "trend"]] = np.nan
    return out


def _bollinger(s: pd.Series, length: int, mult: float) -> Tuple[pd.Series, pd.Series]:
    basis = s.rolling(length).mean()
    dev = s.rolling(length).std(ddof=0) * mult
    return basis + dev, basis - dev


def warmup_bars(cfg: Config) -> int:
    """History window including prior crossovers, beyond the indicator seed length."""
    w = cfg.rsi_period * 2 - 1
    base = cfg.rsi_period + cfg.sf + 2 * w
    if cfg.strategy == "qqe_mod":
        w2 = cfg.rsi_period2 * 2 - 1
        base = max(base, cfg.rsi_period2 + cfg.sf2 + 2 * w2) + cfg.bb_length
    return max(600, base * 4)


def resample_ohlcv(df_1h: pd.DataFrame, timeframe: str) -> pd.DataFrame:
    """Discard bars missing any constituent hour."""
    hours = tf_hours(timeframe)
    df = df_1h.sort_index()
    if hours == 1:
        return df
    agg = df.resample(f"{hours}h", label="left", closed="left").agg(
        {"open": "first", "high": "max", "low": "min", "close": "last", "volume": "sum"})
    counts = df["close"].resample(f"{hours}h", label="left", closed="left").count()
    return agg[counts == hours].dropna()


def compute_signals(df: pd.DataFrame, cfg: Config) -> pd.DataFrame:
    """Return close, entry, exit and indicators; signals are available after bar close."""
    close = df["close"].astype(float)
    q1 = qqe(close, cfg.rsi_period, cfg.sf, cfg.factor)
    out = pd.DataFrame({"close": close}, index=df.index)
    out["rsi_ma"] = q1["rsi_ma"]
    out["fast_tl"] = q1["fast_tl"]
    out["trend"] = q1["trend"]

    if cfg.strategy == "qqe":
        prev = q1["trend"].shift(1)
        out["entry"] = (q1["trend"] == 1) & (prev == -1)
        out["exit"] = (q1["trend"] == -1) & (prev == 1)
    else:
        q2 = qqe(close, cfg.rsi_period2, cfg.sf2, cfg.factor2)
        upper, lower = _bollinger(q1["fast_tl"] - 50.0, cfg.bb_length, cfg.bb_mult)
        hist1 = q1["rsi_ma"] - 50.0
        hist2 = q2["rsi_ma"] - 50.0
        green = (hist2 > cfg.threshold) & (hist1 > upper)
        red = (hist2 < -cfg.threshold) & (hist1 < lower)
        out["green"] = green
        out["red"] = red
        out["entry"] = green & ~green.shift(1, fill_value=False)
        out["exit"] = (~green & green.shift(1, fill_value=False)) | red

    # NaN indicator comparisons already suppress signals; do not truncate manually.
    # Indicator warmup differs from required live history, which warmup_bars defines.
    out["entry"] = out["entry"].fillna(False).astype(bool)
    out["exit"] = out["exit"].fillna(False).astype(bool)
    return out


def stop_loss_hit(entry_price: float, price: float, stop_loss_pct: float) -> bool:
    """Zero stop_loss_pct disables stops."""
    if stop_loss_pct <= 0 or entry_price <= 0 or price <= 0:
        return False
    return price <= entry_price * (1.0 - stop_loss_pct / 100.0)


def decide_last_bar(df: pd.DataFrame, cfg: Config, currently_long: bool,
                    entry_price: float = 0.0) -> Tuple[Optional[str], str]:
    sig = compute_signals(df, cfg)
    if len(sig) == 0:
        return None, "No data"
    last = sig.iloc[-1]
    if currently_long:
        if stop_loss_hit(entry_price, float(last["close"]), cfg.stop_loss_pct):
            return "SELL", REASON_STOP
        if bool(last["exit"]):
            return "SELL", REASON_EXIT
        return None, "Hold position"
    if bool(last["entry"]):
        return "BUY", REASON_ENTRY
    return None, "Wait"
