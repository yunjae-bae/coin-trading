# -*- coding: utf-8 -*-
from __future__ import annotations

import argparse
import os
from typing import Dict, List, Optional

import pandas as pd

from coin_config import BASE_DIR, Config, describe, load_config
from coin_kernel import (REASON_EXIT, REASON_STOP, compute_signals, resample_ohlcv,
                         stop_loss_hit, warmup_bars)

OUT_DIR = os.path.join(BASE_DIR, "coin_backtest_output")


def simulate(bars: pd.DataFrame, sig: pd.DataFrame, cfg: Config, init_cash: float,
             collect_rows: bool = False) -> Dict:
    """Fill close signals at the next open; retain earlier history for warmup."""
    opens = bars["open"].astype(float)
    idx = list(sig.index)
    warm = min(warmup_bars(cfg), max(0, len(idx) - 2))

    krw = float(init_cash)
    qty = 0.0
    entry_price = 0.0
    fees = 0.0
    n_stop = 0
    trades: List[dict] = []
    equity: List[dict] = []
    wins = 0
    round_trips = 0

    lows = bars["low"].astype(float)
    last_exit_i = -10 ** 9
    for i in range(warm, len(idx)):
        ts = idx[i]
        close = float(sig.at[ts, "close"])
        if qty > 0 and cfg.stop_loss_pct > 0:
            stop_price = entry_price * (1.0 - cfg.stop_loss_pct / 100.0)
            if float(lows.loc[ts]) <= stop_price:
                fill = min(stop_price, float(opens.loc[ts])) * (1.0 - cfg.slippage_pct / 100.0)
                gross = qty * fill
                fee = gross * cfg.fee_rate
                krw += gross - fee
                fees += fee
                round_trips += 1
                n_stop += 1
                if fill > entry_price:
                    wins += 1
                if collect_rows:
                    trades.append({"ts": ts, "side": "SELL", "reason": REASON_STOP,
                                   "price": fill, "qty": qty, "fee_krw": fee, "krw_after": krw})
                qty = 0.0
                entry_price = 0.0
                last_exit_i = i

        action: Optional[str] = None
        reason = ""
        if qty > 0:
            if bool(sig.at[ts, "exit"]):
                action, reason = "SELL", REASON_EXIT
        elif bool(sig.at[ts, "entry"]) and (i - last_exit_i) > cfg.cooldown_bars:
            action, reason = "BUY", "entry"

        if action and i + 1 < len(idx):
            raw_px = float(opens.loc[idx[i + 1]])
            slip = cfg.slippage_pct / 100.0
            px = raw_px * (1.0 + slip) if action == "BUY" else raw_px * (1.0 - slip)
            if action == "BUY":
                budget = krw * cfg.position_pct / 100.0
                order_krw = budget / (1.0 + cfg.fee_rate)
                if order_krw >= cfg.min_order_krw:
                    fee = order_krw * cfg.fee_rate
                    qty = order_krw / px
                    krw -= order_krw + fee
                    fees += fee
                    entry_price = px
                    if collect_rows:
                        trades.append({"ts": idx[i + 1], "side": "BUY", "reason": reason,
                                       "price": px, "qty": qty, "fee_krw": fee, "krw_after": krw})
            else:
                gross = qty * px
                fee = gross * cfg.fee_rate
                krw += gross - fee
                fees += fee
                round_trips += 1
                if px > entry_price:
                    wins += 1
                if reason == REASON_STOP:
                    n_stop += 1
                if collect_rows:
                    trades.append({"ts": idx[i + 1], "side": "SELL", "reason": reason,
                                   "price": px, "qty": qty, "fee_krw": fee, "krw_after": krw})
                qty = 0.0
                entry_price = 0.0
                last_exit_i = i
        equity.append({"ts": ts, "equity_krw": krw + qty * close})

    eq = pd.DataFrame(equity)
    final = float(eq["equity_krw"].iloc[-1]) if len(eq) else init_cash
    mdd = float(((eq["equity_krw"] / eq["equity_krw"].cummax()) - 1.0).min()) * 100.0 if len(eq) else 0.0
    first_close = float(sig["close"].iloc[warm])
    last_close = float(sig["close"].iloc[-1])
    return {
        "ret_pct": (final / init_cash - 1.0) * 100.0,
        "final": final,
        "mdd_pct": mdd,
        "bh_pct": (last_close / first_close - 1.0) * 100.0,
        "round_trips": round_trips,
        "win_rate": (wins / round_trips * 100.0) if round_trips else 0.0,
        "stops": n_stop,
        "fees": fees,
        "bars": len(eq),
        "slippage_pct": cfg.slippage_pct,
        "start": sig.index[warm],
        "end": sig.index[-1],
        "trades": trades,
        "equity": eq,
    }


def load_bars(path: str) -> pd.DataFrame:
    if not os.path.exists(path):
        raise SystemExit(f"Data not found: {path}\nRun upbit_candle_fetcher.py first.")
    df = pd.read_parquet(path)
    df.index = pd.to_datetime(df.index, utc=True)
    return df.sort_index()


def mod_cfg(**kw) -> Config:
    return Config(strategy="qqe_mod", rsi_period=6, factor=3.0, **kw)


def compare(df_1h: pd.DataFrame, init_cash: float) -> None:
    combos = []
    for strat in ("qqe", "qqe_mod"):
        for tf in ("1h", "4h"):
            base = mod_cfg(timeframe=tf, position_pct=100.0) if strat == "qqe_mod" \
                else Config(strategy="qqe", timeframe=tf, position_pct=100.0)
            bars = resample_ohlcv(df_1h, tf)
            sig = compute_signals(bars, base)
            for sl in (0.0, 3.0, 5.0, 10.0):
                cfg = Config(**{**base.__dict__, "stop_loss_pct": sl})
                r = simulate(bars, sig, cfg, init_cash)
                combos.append((strat, tf, sl, r))

    slip = combos[0][3]["slippage_pct"]
    print(f"\n=== KRW-BTC QQE Backtest comparison (Initial {init_cash:,.0f} KRW, Allocation 100%, "
          f"Fees 0.05%, Slippage {slip:g}%) ===")
    r0 = combos[0][3]
    print(f"Period: {r0['start']} ~ {r0['end']}   |   Buy and hold(B&H) {r0['bh_pct']:+.2f}%\n")
    print(f"{'strategy':<9}{'TF':<5}{'Stop':>6}{'return':>11}{'MDD':>10}{'Trades':>7}{'Win %':>8}{'stops':>7}{'Fees':>13}")
    print("-" * 80)
    for strat, tf, sl, r in combos:
        sl_s = "None" if sl == 0 else f"{sl:g}%"
        print(f"{strat:<9}{tf:<5}{sl_s:>6}{r['ret_pct']:>10.2f}%{r['mdd_pct']:>9.2f}%"
              f"{r['round_trips']:>7}{r['win_rate']:>7.1f}%{r['stops']:>7}{r['fees']:>12,.0f} KRW")
    best = max(combos, key=lambda c: c[3]["ret_pct"])
    print(f"\nBest return: {best[0]} / {best[1]} / stop loss "
          f"{'None' if best[2] == 0 else f'{best[2]:g}%'} -> {best[3]['ret_pct']:+.2f}%")
    print("Results cover one historical period; selecting the best combination can overfit.")


def single(df_1h: pd.DataFrame, cfg: Config, init_cash: float) -> None:
    bars = resample_ohlcv(df_1h, cfg.timeframe)
    sig = compute_signals(bars, cfg)
    r = simulate(bars, sig, cfg, init_cash, collect_rows=True)

    os.makedirs(OUT_DIR, exist_ok=True)
    pd.DataFrame(r["trades"]).to_csv(os.path.join(OUT_DIR, "trades.csv"), index=False)
    r["equity"].to_csv(os.path.join(OUT_DIR, "equity.csv"), index=False)

    print(f"\n=== {cfg.market} Backtest ===")
    print(describe(cfg))
    print(f"\nPeriod          : {r['start']} ~ {r['end']}  ({r['bars']} bars)")
    print(f"Initial capital     : {init_cash:,.0f} KRW")
    print(f"Final equity     : {r['final']:,.0f} KRW  ({r['ret_pct']:+.2f}%)")
    print(f"Maximum drawdown(MDD): {r['mdd_pct']:.2f}%   |   Buy and hold(B&H): {r['bh_pct']:+.2f}%")
    print(f"Round trips     : {r['round_trips']} trades (win rate {r['win_rate']:.1f}%, stop loss {r['stops']} trades)")
    print(f"Total Fees   : {r['fees']:,.0f} KRW")
    print(f"Output          : {OUT_DIR}/trades.csv, equity.csv")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="")
    ap.add_argument("--cash", type=float, default=10_000_000.0)
    ap.add_argument("--compare", action="store_true", help="Compare strategy, timeframe, and stop combinations")
    args = ap.parse_args()

    cfg = load_config()
    path = args.data or os.path.join(BASE_DIR, "coin_hist_1h", f"{cfg.market}.parquet")
    df_1h = load_bars(path)
    if args.compare:
        compare(df_1h, args.cash)
    else:
        single(df_1h, cfg, args.cash)


if __name__ == "__main__":
    main()
