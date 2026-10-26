from __future__ import annotations

import argparse
import os
from dataclasses import asdict, dataclass, field, replace

import pandas as pd

from coin_config import (
    BASE_DIR,
    Config,
    describe,
    load_config,
)
from coin_kernel import (
    REASON_EXIT,
    REASON_STOP,
    compute_signals,
    resample_ohlcv,
    warmup_bars,
)

OUT_DIR = os.path.join(BASE_DIR, "coin_backtest_output")
REASON_BUY = "entry"


@dataclass
class Trade:

    ts: pd.Timestamp
    side: str
    reason: str
    price: float
    qty: float
    fee_krw: float
    krw_after: float


@dataclass
class Result:
    ret_pct: float
    final: float
    mdd_pct: float
    bh_pct: float
    round_trips: int
    win_rate: float
    stops: int
    fees: float
    bars: int
    slippage_pct: float
    start: pd.Timestamp
    end: pd.Timestamp
    equity: pd.DataFrame
    trades: list[Trade] = field(default_factory=list)


@dataclass
class _Book:

    krw: float
    qty: float = 0.0
    entry_price: float = 0.0
    fees: float = 0.0
    wins: int = 0
    round_trips: int = 0
    stops: int = 0

    def buy(self, px: float, cfg: Config) -> tuple[float, float] | None:
        """Return quantity and fee, or None below the minimum order."""
        budget = self.krw * cfg.position_pct / 100.0
        if cfg.max_order_krw > 0:
            budget = min(budget, cfg.max_order_krw)
        order_krw = budget / (1.0 + cfg.fee_rate)
        if order_krw < cfg.min_order_krw:
            return None
        fee = order_krw * cfg.fee_rate
        self.qty = order_krw / px
        self.krw -= order_krw + fee
        self.fees += fee
        self.entry_price = px
        return self.qty, fee

    def sell(self, px: float, cfg: Config, *, is_stop: bool) -> tuple[float, float]:
        """Return quantity and fee."""
        sold = self.qty
        gross = sold * px
        fee = gross * cfg.fee_rate
        self.krw += gross - fee
        self.fees += fee
        self.round_trips += 1
        self.stops += int(is_stop)
        self.wins += int(px > self.entry_price)
        self.qty = 0.0
        self.entry_price = 0.0
        return sold, fee


def _stop_fill(entry_price: float, low: float, bar_open: float, cfg: Config) -> float | None:
    """Use the open price when a gap crosses the stop."""
    stop_price = entry_price * (1.0 - cfg.stop_loss_pct / 100.0)
    if low > stop_price:
        return None
    return min(stop_price, bar_open) * (1.0 - cfg.slippage_pct / 100.0)


def _fill_price(raw_px: float, cfg: Config, *, is_buy: bool) -> float:
    slip = cfg.slippage_pct / 100.0
    return raw_px * (1.0 + slip) if is_buy else raw_px * (1.0 - slip)


def _decide(holding: bool, entry: bool, exit_: bool, bars_since_exit: int,
            cooldown_bars: int) -> str:
    """Return an empty string when no order is due."""
    if holding:
        return "SELL" if exit_ else ""
    if entry and (cooldown_bars == 0 or bars_since_exit > cooldown_bars):
        return "BUY"
    return ""


def _evaluation_start(index: pd.Index, cfg: Config, start: pd.Timestamp | None) -> int:
    n = len(index)
    if not n:
        raise ValueError("Empty signals; no bars to simulate")
    warm = min(warmup_bars(cfg), max(0, n - 2))
    if start is not None:
        warm = max(warm, int(index.searchsorted(start)))
    if warm >= n:
        raise ValueError("No bars within the evaluation period")
    return warm


def simulate(bars: pd.DataFrame, sig: pd.DataFrame, cfg: Config, init_cash: float,
             *, start: pd.Timestamp | None = None) -> Result:
    """Fill close signals at the next open; retain earlier history for warmup."""
    if not sig.index.equals(bars.index):
        raise ValueError("bars and sig indexes differ; prices and signals must align")
    n = len(sig.index)
    warm = _evaluation_start(sig.index, cfg, start)
    opens = bars["open"].to_numpy(dtype=float)
    lows = bars["low"].to_numpy(dtype=float)
    closes = sig["close"].to_numpy(dtype=float)
    entries = sig["entry"].to_numpy(dtype=bool)
    exits = sig["exit"].to_numpy(dtype=bool)
    idx = sig.index

    book = _Book(krw=float(init_cash))
    trades: list[Trade] = []
    curve: list[float] = []
    last_exit_i = -(10 ** 9)
    pending = ""

    for i in range(warm, n):
        if pending:
            px = _fill_price(opens[i], cfg, is_buy=pending == "BUY")
            if pending == "BUY":
                filled = book.buy(px, cfg)
                if filled:
                    trades.append(Trade(idx[i], "BUY", REASON_BUY, px, *filled, book.krw))
            else:
                sold, fee = book.sell(px, cfg, is_stop=False)
                trades.append(Trade(idx[i], "SELL", REASON_EXIT, px, sold, fee, book.krw))
                last_exit_i = i
        if book.qty > 0 and cfg.stop_loss_pct > 0:
            fill = _stop_fill(book.entry_price, lows[i], opens[i], cfg)
            if fill is not None:
                sold, fee = book.sell(fill, cfg, is_stop=True)
                trades.append(Trade(idx[i], "SELL", REASON_STOP, fill, sold, fee, book.krw))
                last_exit_i = i

        curve.append(book.krw + book.qty * closes[i])
        pending = _decide(book.qty > 0, bool(entries[i]), bool(exits[i]),
                          i - last_exit_i, cfg.cooldown_bars)

    eq = pd.DataFrame({"ts": idx[warm:n], "equity_krw": curve})
    final = float(eq["equity_krw"].iloc[-1])
    drawdown = (eq["equity_krw"] / eq["equity_krw"].cummax()) - 1.0
    return Result(
        ret_pct=(final / init_cash - 1.0) * 100.0,
        final=final,
        mdd_pct=float(drawdown.min()) * 100.0,
        bh_pct=(float(closes[-1]) / float(closes[warm]) - 1.0) * 100.0,
        round_trips=book.round_trips,
        win_rate=(book.wins / book.round_trips * 100.0) if book.round_trips else 0.0,
        stops=book.stops,
        fees=book.fees,
        bars=len(eq),
        slippage_pct=cfg.slippage_pct,
        start=idx[warm],
        end=idx[-1],
        equity=eq,
        trades=trades,
    )


def load_bars(path: str) -> pd.DataFrame:
    if not os.path.exists(path):
        raise SystemExit(f"Data not found: {path}\nRun upbit_candle_fetcher.py first.")
    df = pd.read_parquet(path)
    df.index = pd.DatetimeIndex(pd.to_datetime(df.index, utc=True))
    return df.sort_index()


def compare(df_1h: pd.DataFrame, init_cash: float) -> None:
    combos: list[tuple[str, str, float, Result]] = []
    inputs: list[tuple[Config, pd.DataFrame, pd.DataFrame]] = []
    for strat in ("qqe", "qqe_mod"):
        for tf in ("1h", "4h", "6h"):
            base = Config(strategy=strat, timeframe=tf, position_pct=100.0)
            bars = resample_ohlcv(df_1h, tf)
            if len(bars) <= warmup_bars(base):
                raise ValueError(f"Insufficient history for a common comparison period ({strat}/{tf})")
            sig = compute_signals(bars, base)
            inputs.append((base, bars, sig))
    start = max(bars.index[warmup_bars(cfg)] for cfg, bars, _ in inputs)
    end = min(bars.index[-1] + pd.Timedelta(hours=cfg.bar_hours) for cfg, bars, _ in inputs)
    for base, bars, sig in inputs:
        complete = bars.index + pd.Timedelta(hours=base.bar_hours) <= end
        for sl in (0.0, 3.0, 5.0, 10.0):
            r = simulate(bars.loc[complete], sig.loc[complete], replace(base, stop_loss_pct=sl),
                         init_cash, start=start)
            combos.append((base.strategy, base.timeframe, sl, r))

    r0 = combos[0][3]
    print(f"\n=== KRW-BTC QQE Backtest comparison (Initial {init_cash:,.0f} KRW, Allocation 100%, "
          f"Fees 0.05%, Slippage {r0.slippage_pct:g}%) ===")
    print(f"Period: {start} ~ {end} (end exclusive)\n")
    print(f"{'strategy':<9}{'TF':<5}{'Stop':>6}{'return':>11}{'MDD':>10}{'Trades':>7}{'Win %':>8}{'stops':>7}{'Fees':>13}")
    print("-" * 80)
    for strat, tf, sl, r in combos:
        sl_s = "None" if sl == 0 else f"{sl:g}%"
        print(f"{strat:<9}{tf:<5}{sl_s:>6}{r.ret_pct:>10.2f}%{r.mdd_pct:>9.2f}%"
              f"{r.round_trips:>7}{r.win_rate:>7.1f}%{r.stops:>7}{r.fees:>12,.0f} KRW")
    best = max(combos, key=lambda c: c[3].ret_pct)
    print(f"\nBest return: {best[0]} / {best[1]} / stop loss "
          f"{'None' if best[2] == 0 else f'{best[2]:g}%'} -> {best[3].ret_pct:+.2f}%")
    print("Results cover one historical period; selecting the best combination can overfit.")


def single(df_1h: pd.DataFrame, cfg: Config, init_cash: float) -> None:
    bars = resample_ohlcv(df_1h, cfg.timeframe)
    sig = compute_signals(bars, cfg)
    r = simulate(bars, sig, cfg, init_cash)

    os.makedirs(OUT_DIR, exist_ok=True)
    pd.DataFrame([asdict(t) for t in r.trades]).to_csv(
        os.path.join(OUT_DIR, "trades.csv"), index=False)
    r.equity.to_csv(os.path.join(OUT_DIR, "equity.csv"), index=False)

    print(f"\n=== {cfg.market} Backtest ===")
    print(describe(cfg))
    print(f"\nPeriod          : {r.start} ~ {r.end}  ({r.bars} bars)")
    print(f"Initial capital     : {init_cash:,.0f} KRW")
    print(f"Final equity     : {r.final:,.0f} KRW  ({r.ret_pct:+.2f}%)")
    print(f"Maximum drawdown(MDD): {r.mdd_pct:.2f}%   |   Buy and hold(B&H): {r.bh_pct:+.2f}%")
    print(f"Round trips     : {r.round_trips} trades (win rate {r.win_rate:.1f}%, stop loss {r.stops} trades)")
    print(f"Total Fees   : {r.fees:,.0f} KRW")
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
