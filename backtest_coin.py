# -*- coding: utf-8 -*-
from __future__ import annotations

import argparse
import os
from typing import List

import pandas as pd

from coin_kernel import FEE_RATE, MARKET, MIN_ORDER_KRW, target_actions

OUT_DIR = "coin_backtest_output"


def run_backtest(df_1h: pd.DataFrame, init_cash: float) -> None:
    lines = target_actions(df_1h)
    opens = df_1h["open"].astype(float)

    krw = float(init_cash)
    btc = 0.0
    trades: List[dict] = []
    equity_rows: List[dict] = []

    idx = list(lines.index)
    for i, ts in enumerate(idx):
        action = lines.at[ts, "action"]
        if action and i + 1 < len(idx):
            fill_ts = idx[i + 1]
            px = float(opens.loc[fill_ts])
            if action == "BUY" and krw >= MIN_ORDER_KRW:
                order_krw = krw / (1.0 + FEE_RATE)
                qty = order_krw / px
                fee = order_krw * FEE_RATE
                btc += qty
                krw -= order_krw + fee
                trades.append({"ts": fill_ts, "side": "BUY", "price": px, "qty": qty,
                               "fee_krw": fee, "krw_after": krw, "btc_after": btc})
            elif action == "SELL" and btc > 0:
                gross = btc * px
                fee = gross * FEE_RATE
                krw += gross - fee
                trades.append({"ts": fill_ts, "side": "SELL", "price": px, "qty": btc,
                               "fee_krw": fee, "krw_after": krw, "btc_after": 0.0})
                btc = 0.0
        mark = float(lines.at[ts, "close_1h"])
        equity_rows.append({"ts": ts, "equity_krw": krw + btc * mark, "krw": krw, "btc": btc})

    os.makedirs(OUT_DIR, exist_ok=True)
    trades_df = pd.DataFrame(trades)
    equity_df = pd.DataFrame(equity_rows)
    trades_df.to_csv(os.path.join(OUT_DIR, "trades.csv"), index=False)
    equity_df.to_csv(os.path.join(OUT_DIR, "equity.csv"), index=False)

    final_mark = float(lines["close_1h"].iloc[-1]) if len(lines) else 0.0
    final_equity = krw + btc * final_mark
    n_round = sum(1 for t in trades if t["side"] == "SELL")
    meets = int(lines["meet"].sum()) if len(lines) else 0
    bh_ret = (final_mark / float(lines["close_1h"].iloc[0]) - 1.0) * 100.0 if len(lines) else 0.0
    dd = 0.0
    if len(equity_df):
        eq = equity_df["equity_krw"]
        dd = float(((eq / eq.cummax()) - 1.0).min()) * 100.0

    print(f"\n=== {MARKET} 1h/4h crossover backtest ===")
    print(f"Period          : {lines.index.min()} ~ {lines.index.max()}" if len(lines) else "Period: (no data)")
    print(f"Crossover/touch count: {meets}  |  Filled trades: {len(trades)} (round trips {n_round} trades)")
    print(f"Initial capital     : {init_cash:,.0f} KRW")
    print(f"Final equity     : {final_equity:,.0f} KRW  ({(final_equity / init_cash - 1.0) * 100.0:+.2f}%)")
    print(f"Maximum drawdown(MDD): {dd:.2f}%  |  Buy and hold(B&H): {bh_ret:+.2f}%")
    print(f"Output          : {OUT_DIR}/trades.csv, equity.csv")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default=os.path.join("coin_hist_1h", f"{MARKET}.parquet"))
    ap.add_argument("--cash", type=float, default=10_000_000.0)
    args = ap.parse_args()

    if not os.path.exists(args.data):
        raise SystemExit(f"Data not found: {args.data}; run upbit_candle_fetcher.py first.")
    df = pd.read_parquet(args.data)
    df.index = pd.to_datetime(df.index, utc=True)
    df = df.sort_index()
    run_backtest(df, args.cash)


if __name__ == "__main__":
    main()
