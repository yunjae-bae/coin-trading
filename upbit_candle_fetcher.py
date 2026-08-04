#!/usr/bin/env python3
# -*- coding: utf-8 -*-
from __future__ import annotations

import argparse
import os
import time
from datetime import datetime, timedelta, timezone
from typing import Any, Dict, List, Optional

import pandas as pd
import requests

BASE_URL = "https://api.upbit.com/v1"
SLEEP_SEC = 0.15
MAX_RETRY = 5


def fetch_page(market: str, to_utc: Optional[str], count: int = 200) -> List[Dict[str, Any]]:
    params: Dict[str, Any] = {"market": market, "count": count}
    if to_utc:
        params["to"] = to_utc
    for attempt in range(MAX_RETRY):
        r = requests.get(f"{BASE_URL}/candles/minutes/60", params=params, timeout=10)
        if r.status_code == 429:
            time.sleep(1.0 * (attempt + 1))
            continue
        r.raise_for_status()
        return r.json()
    raise RuntimeError(f"{market}: {MAX_RETRY} consecutive HTTP 429 responses; retry later")


def rows_to_df(rows: List[Dict[str, Any]]) -> pd.DataFrame:
    if not rows:
        return pd.DataFrame(columns=["open", "high", "low", "close", "volume"])
    df = pd.DataFrame(rows)
    df["ts"] = pd.to_datetime(df["candle_date_time_utc"], errors="coerce").dt.tz_localize(timezone.utc)
    df = df.rename(columns={"opening_price": "open", "high_price": "high", "low_price": "low",
                            "trade_price": "close", "candle_acc_trade_volume": "volume"})
    df = df.dropna(subset=["ts"]).set_index("ts").sort_index()
    return df[["open", "high", "low", "close", "volume"]].astype(float)


def fetch_history(market: str, start_utc: datetime, stop_at: Optional[datetime] = None) -> pd.DataFrame:
    """Fetch backwards to start_utc; stop at stop_at when updating existing data."""
    frames: List[pd.DataFrame] = []
    to: Optional[str] = None
    total = 0
    while True:
        rows = fetch_page(market, to)
        if not rows:
            break
        df = rows_to_df(rows)
        frames.append(df)
        total += len(df)
        oldest = df.index.min()
        print(f"[..] {market} {total} bars collected, oldest={oldest.isoformat()}")
        if oldest <= start_utc or (stop_at is not None and oldest <= stop_at) or len(rows) < 200:
            break
        to = oldest.strftime("%Y-%m-%dT%H:%M:%SZ")  # The exclusive to boundary excludes the oldest bar from the next page.
        time.sleep(SLEEP_SEC)
    if not frames:
        return rows_to_df([])
    out = pd.concat(frames)
    out = out[~out.index.duplicated(keep="last")].sort_index()
    return out[out.index >= start_utc]


def main() -> None:
    ap = argparse.ArgumentParser(description="Fetch Upbit hourly candles to Parquet")
    ap.add_argument("--market", default="KRW-BTC")
    ap.add_argument("--days", type=int, default=365, help="Days of history to collect initially")
    ap.add_argument("--outdir", default="coin_hist_1h")
    ap.add_argument("--update", action="store_true", help="Fetch only bars newer than the existing Parquet data")
    args = ap.parse_args()

    os.makedirs(args.outdir, exist_ok=True)
    path = os.path.join(args.outdir, f"{args.market}.parquet")

    now_utc = datetime.now(timezone.utc)
    start_utc = now_utc - timedelta(days=args.days)

    existing: Optional[pd.DataFrame] = None
    stop_at: Optional[datetime] = None
    if args.update and os.path.exists(path):
        existing = pd.read_parquet(path)
        existing.index = pd.to_datetime(existing.index, utc=True)
        if len(existing):
            stop_at = existing.index.max().to_pydatetime()
            print(f"[OK] Existing {len(existing)} bars, latest={stop_at.isoformat()} ; fetching newer bars")

    df_new = fetch_history(args.market, start_utc, stop_at=stop_at)

    if existing is not None and len(existing):
        df_all = pd.concat([existing, df_new])
        df_all = df_all[~df_all.index.duplicated(keep="last")].sort_index()
    else:
        df_all = df_new

    df_all = df_all[df_all.index + pd.Timedelta(hours=1) <= pd.Timestamp(now_utc)]

    df_all.to_parquet(path)
    print(f"[OK] {args.market}: New {len(df_new)} bars / total {len(df_all)} bars -> {path}")
    if len(df_all):
        print(f"[OK] Range: {df_all.index.min().isoformat()} ~ {df_all.index.max().isoformat()}")


if __name__ == "__main__":
    main()
