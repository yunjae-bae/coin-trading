# -*- coding: utf-8 -*-
from __future__ import annotations

from typing import Optional

import numpy as np
import pandas as pd

MARKET = "KRW-BTC"
FEE_RATE = 0.0005
MIN_ORDER_KRW = 5000.0
WARMUP_HOURS = 9


def build_lines(df_1h: pd.DataFrame) -> pd.DataFrame:
    if df_1h is None or len(df_1h) == 0:
        return pd.DataFrame(columns=["close_1h", "line_4h", "diff"])
    df = df_1h.sort_index()
    close_1h = df["close"].astype(float)

    close_4h = close_1h.resample("4h", label="left", closed="left").last().dropna()
    known_at = close_4h.copy()
    known_at.index = known_at.index + pd.Timedelta(hours=4)
    line_4h = known_at.reindex(close_1h.index, method="ffill")

    out = pd.DataFrame({"close_1h": close_1h, "line_4h": line_4h})
    out = out.dropna()
    out["diff"] = out["close_1h"] - out["line_4h"]
    return out


def meet_events(lines: pd.DataFrame) -> pd.Series:
    d = lines["diff"].astype(float)
    prev = d.shift(1)
    crossed = (np.sign(d) * np.sign(prev)) < 0
    touched = (d == 0) & (prev != 0)
    ev = (crossed | touched) & prev.notna()
    return ev.fillna(False)


def target_actions(df_1h: pd.DataFrame) -> pd.DataFrame:
    lines = build_lines(df_1h)
    if lines.empty:
        lines["meet"] = pd.Series(dtype=bool)
        lines["action"] = pd.Series(dtype=str)
        return lines
    lines = lines.copy()
    lines["meet"] = meet_events(lines)
    actions = []
    long_now = False
    for meet in lines["meet"]:
        if meet and not long_now:
            actions.append("BUY")
            long_now = True
        elif meet and long_now:
            actions.append("SELL")
            long_now = False
        else:
            actions.append("")
    lines["action"] = actions
    return lines


def decide_last_bar(df_1h: pd.DataFrame, currently_long: bool) -> Optional[str]:
    lines = build_lines(df_1h)
    if len(lines) < 2:
        return None
    if not bool(meet_events(lines).iloc[-1]):
        return None
    return "SELL" if currently_long else "BUY"
