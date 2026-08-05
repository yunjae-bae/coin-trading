# -*- coding: utf-8 -*-
from __future__ import annotations

import csv
import os
import sys
import time
import traceback
from datetime import datetime, timedelta, timezone
from typing import Optional, Tuple

import pandas as pd

import notify
from bot_state import KST, StateStore, StopFlag, fmt, now_kst
from coin_config import BASE_DIR, ENV_PATH, Config, ConfigError, describe, load_config
from coin_kernel import (REASON_STOP, compute_signals, decide_last_bar, resample_ohlcv,
                         stop_loss_hit, warmup_bars)
from upbit_api import Upbit, UpbitError

OUT_DIR = os.path.join(BASE_DIR, "out")
LOG_PATH = os.path.join(OUT_DIR, "\uae30\ub85d.log")
ORDERS_PATH = os.path.join(OUT_DIR, "\uc8fc\ubb38\ub0b4\uc5ed.csv")
STATUS_PATH = os.path.join(OUT_DIR, "status.txt")
STATE_PATH = os.path.join(OUT_DIR, "state.json")
STOP_PATH = os.path.join(OUT_DIR, "\ubd07\uc815\uc9c0.txt")

ORDER_HEADER = ["timestamp", "side", "reason", "market", "quantity", "fill_price", "amount", "mode", "order_id"]
LOG_MAX_BYTES = 5 * 1024 * 1024


def _rotate_log() -> None:
    try:
        if os.path.getsize(LOG_PATH) < LOG_MAX_BYTES:
            return
        os.replace(LOG_PATH, LOG_PATH + ".1")
    except OSError:
        pass


def log(msg: str) -> None:
    os.makedirs(OUT_DIR, exist_ok=True)
    line = f"[{fmt(now_kst())}] {msg}"
    print(line, flush=True)
    try:
        _rotate_log()
        with open(LOG_PATH, "a", encoding="utf-8", newline="\n") as f:
            f.write(line + "\n")
    except OSError:
        pass


def log_order(row: dict) -> None:
    os.makedirs(OUT_DIR, exist_ok=True)
    new = not (os.path.exists(ORDERS_PATH) and os.path.getsize(ORDERS_PATH) > 0)
    with open(ORDERS_PATH, "a", newline="", encoding="utf-8-sig") as f:   # UTF-8 BOM for spreadsheet compatibility.
        w = csv.DictWriter(f, fieldnames=ORDER_HEADER)
        if new:
            w.writeheader()
        w.writerow({k: row.get(k, "") for k in ORDER_HEADER})


class Bot:
    def __init__(self, cfg: Config):
        self.cfg = cfg
        self.api = Upbit(cfg.access_key, cfg.secret_key)
        self.store = StateStore(STATE_PATH)
        self.stop = StopFlag(STOP_PATH)
        self.last_error = ""
        self.last_decision = "-"
        self.fail_streak = 0
        self.next_wake: Optional[datetime] = None

    def alert(self, text: str) -> None:
        try:
            tag = "" if not self.cfg.dry_run else "[dry run] "
            notify.send(self.cfg, tag + text)
        except Exception as e:
            log(f"  (Notification delivery failed: {type(e).__name__})")


    def fetch_bars(self) -> pd.DataFrame:
        need_1h = (warmup_bars(self.cfg) + 5) * self.cfg.bar_hours
        df_1h = self.api.candles_1h(self.cfg.market, need_1h)
        bars = resample_ohlcv(df_1h, self.cfg.timeframe)
        if len(bars) < warmup_bars(self.cfg):
            raise UpbitError(
                f"Insufficient historical bars for a decision ({len(bars)} / {warmup_bars(self.cfg)} required)",
                "Retrying shortly.")
        return bars

    def read_account(self, price: Optional[float]) -> Tuple[float, float]:
        if self.cfg.dry_run and not (self.cfg.access_key and self.cfg.secret_key):
            pos = self.store.state.position
            return 10_000_000.0, pos.qty
        return self.api.balances(self.cfg.market)


    def _identifier(self, bar_iso: str, side: str) -> str:
        return f"qqe-{bar_iso.replace('-', '').replace(':', '')[:13]}-{side}"

    def buy(self, krw: float, price: Optional[float], bar_iso: str, reason: str,
            bar_close: float = 0.0) -> None:
        budget = krw * self.cfg.position_pct / 100.0
        if self.cfg.max_order_krw > 0:
            budget = min(budget, self.cfg.max_order_krw)
        order_krw = float(int(budget / (1.0 + self.cfg.fee_rate)))
        if order_krw < self.cfg.min_order_krw:
            log(f"  Skipping buy: available order amount {order_krw:,.0f} KRW is below the minimum order amount "
                f"{self.cfg.min_order_krw:,.0f} KRW")
            return
        # Do not guard sells: delaying a risk-reducing exit can be more dangerous.
        if price and bar_close and self.cfg.price_guard_pct > 0:
            gap = (price / bar_close - 1.0) * 100.0
            if abs(gap) > self.cfg.price_guard_pct:
                log(f"  Buy deferred: signal price ({bar_close:,.0f} KRW), current price differs by {gap:+.2f}% "
                    f"(allowed {self.cfg.price_guard_pct:g}%)")
                return
        ident = self._identifier(bar_iso, "bid")
        if self.cfg.dry_run:
            log(f"  [dry run] buy {order_krw:,.0f} KRW (approximate price {price:,.0f} KRW)" if price
                else f"  [dry run] buy {order_krw:,.0f} KRW")
            qty = order_krw / price if price else 0.0
            self.store.open_position(qty, price or 0.0)
            self.alert(f"🟢 buy\n{self.cfg.market} {order_krw:,.0f} KRW\n"
                       f"Price {price:,.0f} KRW\nReason: {reason}" if price else
                       f"🟢 buy\n{self.cfg.market} {order_krw:,.0f} KRW\nReason: {reason}")
            log_order({"timestamp": fmt(now_kst()), "side": "buy", "reason": reason,
                       "market": self.cfg.market, "quantity": f"{qty:.8f}", "fill_price": price or "",
                       "amount": f"{order_krw:.0f}", "mode": "dry run", "order_id": ident})
            return

        qty_before = self.api.balances(self.cfg.market)[1]
        res = self.api.order({"market": self.cfg.market, "side": "bid", "ord_type": "price",
                              "price": str(order_krw), "identifier": ident})
        state, exec_qty, avg = self.api.wait_done(ident)
        if exec_qty <= 0:
            time.sleep(2.0)
            exec_qty = max(0.0, self.api.balances(self.cfg.market)[1] - qty_before)
            avg = price or 0.0
        if exec_qty > 0:
            self.store.open_position(exec_qty, avg or price or 0.0)
            log(f"  Buy filled {exec_qty:.8f} {self.cfg.market.split('-')[1]} "
                f"@ {avg:,.0f} KRW (status {state})")
            sl = (avg or price or 0.0) * (1 - self.cfg.stop_loss_pct / 100)
            self.alert(f"🟢 Buy filled\n{self.cfg.market} {exec_qty:.8f}\n"
                       f"Fill price {avg:,.0f} KRW ({order_krw:,.0f} KRW total)\n"
                       + (f"Stop level {sl:,.0f} KRW\n" if self.cfg.stop_loss_pct > 0 else "")
                       + f"Reason: {reason}")
        else:
            log(f"  ⚠ Buy submitted but fill unconfirmed (status {state}). "
                f"The balance will be checked again next cycle.")
            self.alert(f"⚠️ Buy submitted but fill unconfirmed\n"
                       f"{self.cfg.market} {order_krw:,.0f} KRW (status {state})\n"
                       f"Check the order history in the Upbit app.")
        log_order({"timestamp": fmt(now_kst()), "side": "buy", "reason": reason, "market": self.cfg.market,
                   "quantity": f"{exec_qty:.8f}", "fill_price": f"{avg:.0f}" if avg else "",
                   "amount": f"{order_krw:.0f}",
                   "mode": "live" if exec_qty > 0 else "live (unconfirmed)",
                   "order_id": res.get("uuid") or ident})

    def sell(self, price: Optional[float], bar_iso: str, reason: str) -> None:
        pos = self.store.state.position
        qty = int(pos.qty * 1e8) / 1e8          # Truncate to eight decimals; rounding up could exceed the balance.
        if qty <= 0:
            log("  Skipping sell: no bot-owned position")
            self.store.close_position()
            return
        # Unsellable dust would otherwise trigger retries on every bar.
        if price and qty * price < self.cfg.min_order_krw:
            log(f"  ⚠ Remaining quantity {qty:.8f} is valued at {qty * price:,.0f} KRW, below the minimum order amount"
                f"({self.cfg.min_order_krw:,.0f} KRW); unable to sell.\n"
                f"     → The bot abandons this dust position and waits for the next signal. "
                f"You can manage it directly in the Upbit app.")
            self.store.close_position()
            return
        ident = self._identifier(bar_iso, "ask")
        if self.cfg.dry_run:
            log(f"  [dry run] sell {qty:.8f} (approximate price {price:,.0f} KRW)" if price
                else f"  [dry run] sell {qty:.8f}")
            pnl = ((price / pos.entry_price - 1) * 100) if (price and pos.entry_price) else None
            self.store.close_position()
            self.alert(f"🔴 sell\n{self.cfg.market} {qty:.8f}\n"
                       + (f"Price {price:,.0f} KRW ({pnl:+.2f}%)\n" if pnl is not None else "")
                       + f"Reason: {reason}")
            log_order({"timestamp": fmt(now_kst()), "side": "sell", "reason": reason,
                       "market": self.cfg.market, "quantity": f"{qty:.8f}", "fill_price": price or "",
                       "amount": f"{qty * (price or 0):.0f}", "mode": "dry run", "order_id": ident})
            return

        try:
            res = self.api.order({"market": self.cfg.market, "side": "ask", "ord_type": "market",
                                  "volume": f"{qty:.8f}", "identifier": ident})
        except UpbitError as e:
            if e.code in ("under_min_total_ask", "insufficient_funds_ask"):
                log(f"  ⚠ Upbit rejected the sell order: {e}\n"
                    f"     → Remaining quantity({qty:.8f}) is too small to sell. "
                    f"The bot abandons this dust position and waits for the next signal.")
                self.store.close_position()
                return
            raise
        state, exec_qty, avg = self.api.wait_done(ident)
        remain = self.api.balances(self.cfg.market)[1]
        if remain < pos.qty * 0.001:
            entry = pos.entry_price
            self.store.close_position()
            log(f"  Sell filled {exec_qty or qty:.8f} @ {avg:,.0f} KRW (status {state})")
            pnl = ((avg / entry - 1) * 100) if (avg and entry) else None
            self.alert(f"🔴 Sell filled\n{self.cfg.market} {exec_qty or qty:.8f}\n"
                       f"Fill price {avg:,.0f} KRW"
                       + (f" ({pnl:+.2f}%)\n" if pnl is not None else "\n")
                       + f"Reason: {reason}")
        else:
            self.store.mark_pending_exit(min(pos.qty, remain))
            log(f"  ⚠ Sell partially filled (Remaining quantity {remain:.8f}). "
                f"The remaining position will be closed regardless of the next signal.")
            self.alert(f"⚠️ Sell partially filled\n{self.cfg.market}\n"
                       f"Sold quantity {exec_qty or 0:.8f} / Remaining quantity {remain:.8f}\n"
                       f"Retrying the remainder in one minute.")
        log_order({"timestamp": fmt(now_kst()), "side": "sell", "reason": reason, "market": self.cfg.market,
                   "quantity": f"{exec_qty or qty:.8f}", "fill_price": f"{avg:.0f}" if avg else "",
                   "amount": f"{(exec_qty or qty) * (avg or price or 0):.0f}", "mode": "live",
                   "order_id": res.get("uuid") or ident})


    def cycle(self) -> None:
        cfg = self.cfg
        bars = self.fetch_bars()
        price = self.api.last_price(cfg.market)
        krw, coin_qty = self.read_account(price)

        warn = self.store.reconcile(coin_qty)
        if warn:
            log(f"  ⚠ {warn}")
            self.alert(f"⚠️ Account balance differs from bot records\n{warn}")

        pos = self.store.state.position
        today = now_kst().strftime("%Y-%m-%d")
        if not pos.is_open and price and coin_qty * price >= cfg.min_order_krw:
            if self.store.state.orphan_notified != today:
                self.store.state.orphan_notified = today
                self.store.save()
                msg = (f"Account contains {coin_qty:.8f} {cfg.market.split('-')[1]}"
                       f"({coin_qty * price:,.0f} KRW) not tracked by the bot.\n"
                       f"Manually acquired holdings are left untouched.\n"
                       f"Investigate if these are bot-owned holdings with missing state.")
                log(f"  ⚠ {msg}")
                self.alert(f"⚠️ Untracked holdings detected\n{msg}")
        bar_iso = bars.index[-1].isoformat()
        close = float(bars["close"].iloc[-1])
        # Use the latest close if the price request fails; entry price would hide losses.
        mark = price or close
        equity = krw + pos.qty * mark
        if self.store.roll_day(equity):
            log(f"  Starting daily reference equity: {equity:,.0f} KRW")

        if cfg.daily_loss_limit_pct > 0:
            loss = self.store.day_loss_pct(equity)
            if loss <= -cfg.daily_loss_limit_pct:
                why = (f"Daily loss {loss:.2f}% exceeds the configured limit ({cfg.daily_loss_limit_pct:g}%)."
                       f"\nStarting equity {self.store.state.day_start_equity:,.0f} KRW "
                       f"→ Current {equity:,.0f} KRW")
                self.stop.raise_flag(why)
                log(f"  ■ Stopping: daily loss limit exceeded\n{why}")
                self.last_error = "Stopped: daily loss limit exceeded (see stop file in out/)"
                self.write_status(mark, krw)
                self.alert(f"🛑 The bot has stopped\n{why}\n\n"
                           f"Review the account, delete the stop file in out/, and restart.")
                raise SystemExit(0)

        if self.store.state.pending_exit and pos.is_open:
            action, reason = "SELL", "Close remaining position"
        else:
            action, reason = decide_last_bar(bars, cfg, pos.is_open, pos.entry_price)
        self.last_decision = f"{reason} ({fmt(bars.index[-1].to_pydatetime())}  bars)"

        bar_end = bars.index[-1].to_pydatetime() + timedelta(hours=cfg.bar_hours)
        age_min = (self.api.now_utc() - bar_end).total_seconds() / 60.0
        if age_min > cfg.bar_hours * 60 + 30:
            log(f"  ⚠ Market data is {age_min / 60:.1f} hours old. Upbit may be under maintenance or "
                f"market data has stopped updating. Skipping this decision.")
            return
        if action == "BUY" and cfg.signal_max_age_min > 0 and age_min > cfg.signal_max_age_min:
            log(f"  Buy deferred: signal bar closed {age_min:.0f} minutes ago "
                f"(allowed {cfg.signal_max_age_min:g} minutes). Stale entry signals are ignored.")
            action = None
            reason = f"Stale signal({age_min:.0f} minutes ago)"
            self.last_decision = f"{reason} — Buy deferred"

        held = f"holding {pos.qty:.8f} (Entry price {pos.entry_price:,.0f} KRW)" if pos.is_open else "No position"
        log(f"   bars {fmt(bars.index[-1].to_pydatetime())} Close {close:,.0f} KRW | {held} | "
            f"KRW balance {krw:,.0f} KRW → {action or 'wait'} ({reason})")

        if not action:
            return
        # Record the attempt before submitting: a lost response followed by restart
        # must not create another buy for the same bar.
        if self.store.state.last_bar_attempted == bar_iso:
            log(f"  Skipping: bar({bar_iso}) was already processed")
            return
        self.store.state.last_bar_attempted = bar_iso
        self.store.save()

        if action == "BUY":
            self.buy(krw, price, bar_iso, reason, close)
        else:
            self.store.want_exit()
            self.sell(price, bar_iso, reason)


    def watch_between_bars(self) -> None:
        st = self.store.state
        pos = st.position
        if not pos.is_open:
            return
        if not st.pending_exit and self.cfg.stop_loss_pct <= 0:
            return
        price = self.api.last_price(self.cfg.market)
        if not price:
            return
        # Keep monitoring identifiers distinct from bar-based identifiers.
        stamp = "w" + now_kst().strftime("%Y%m%d%H%M")
        if st.pending_exit:
            log(f"  Exit incomplete: retrying remaining {pos.qty:.8f} units")
            self.sell(price, stamp, "Close remaining position")
        elif stop_loss_hit(pos.entry_price, price, self.cfg.stop_loss_pct):
            log(f"  ■ Stop level reached — Entry price {pos.entry_price:,.0f} KRW versus current {price:,.0f} KRW "
                f"({(price / pos.entry_price - 1) * 100:+.2f}%). Selling immediately")
            self.store.want_exit()
            self.sell(price, stamp, REASON_STOP)


    def write_status(self, price: Optional[float], krw: Optional[float]) -> None:
        cfg = self.cfg
        pos = self.store.state.position
        lines = [
            "=" * 46,
            "  Bitcoin Trading Bot — Status",
            "=" * 46,
            f"Updated at   : {fmt(now_kst())}",
            f"Mode   : {'Dry run (no orders submitted)' if cfg.dry_run else '★ LIVE ORDERS ★'}",
            f"Market / timeframe   : {cfg.market} / {cfg.timeframe}",
            f"strategy        : {cfg.strategy} | stop loss "
            f"{f'{cfg.stop_loss_pct:g}%' if cfg.stop_loss_pct else 'None'} | "
            f"Allocation {cfg.position_pct:g}%",
            "",
        ]
        if pos.is_open:
            cur = price or 0.0
            pnl = (cur / pos.entry_price - 1.0) * 100.0 if pos.entry_price else 0.0
            lines += [
                f"Position   : Holding  {pos.qty:.8f}",
                f"Entry price      : {pos.entry_price:,.0f} KRW  ({pos.entry_time})",
                f"Current price      : {cur:,.0f} KRW  ({pnl:+.2f}%)",
            ]
            if cfg.stop_loss_pct > 0:
                lines.append(f"Stop level      : {pos.entry_price * (1 - cfg.stop_loss_pct / 100):,.0f} KRW")
        else:
            lines.append("Position   : No position (cash)")
        lines += [
            f"KRW balance   : {krw:,.0f} KRW" if krw is not None else
            "KRW balance   : Unavailable (see error below)",
            "",
            f"Last decision : {self.last_decision}",
            f"Last trade : {self.store.state.last_trade_time or 'None yet'}",
            f"Next check   : {fmt(self.next_wake) if self.next_wake else '-'}",
            "",
            f"Last error   : {self.last_error or 'None'}",
            "",
        ]
        if cfg.dry_run and not (cfg.access_key and cfg.secret_key):
            lines.append("※ Dry run without keys: signals only; actual balances are unchanged.")
        lines += [
            "※ Historical results included frequent losing trades and a few large gains.",
            "※ Losing streaks are possible. Review risk settings in README.md.",
        ]
        try:
            with open(STATUS_PATH, "w", encoding="utf-8") as f:
                f.write("\n".join(lines) + "\n")
        except OSError:
            pass

    def _alert_failure(self, detail: str) -> None:
        n = self.fail_streak
        if n in (3, 10) or (n > 10 and n % 20 == 0):
            self.alert(f"⚠️ Consecutive failures: {n} occurrences\n{detail}\n\n"
                       f"Retrying. Check your connection, API keys, and Upbit service status.")

    def maybe_heartbeat(self, price: Optional[float], krw: Optional[float]) -> None:
        now = now_kst()
        today = now.strftime("%Y-%m-%d")
        if self.store.state.heartbeat_date == today or now.hour < self.cfg.heartbeat_hour:
            return
        self.store.state.heartbeat_date = today
        self.store.save()
        pos = self.store.state.position
        if pos.is_open and price and pos.entry_price:
            held = (f"Holding {pos.qty:.8f}\nEntry price {pos.entry_price:,.0f} KRW → "
                    f"Current {price:,.0f} KRW ({(price / pos.entry_price - 1) * 100:+.2f}%)")
        else:
            held = "No position (cash)"
        self.alert(f"🔔 Trading bot heartbeat\n{fmt(now)}\n\n{held}\n"
                   + (f"KRW balance {krw:,.0f} KRW\n" if krw is not None else "")
                   + f"Last trade: {self.store.state.last_trade_time or 'None yet'}\n"
                   + (f"\n⚠️ Last error: {self.last_error}" if self.last_error else ""))


    def next_bar_close(self) -> datetime:
        """Next bar close plus ten seconds, including a just-closed bar in its grace period."""
        now = self.api.now_utc().astimezone(timezone.utc)
        h = self.cfg.bar_hours
        base = now.replace(minute=0, second=0, microsecond=0)
        cand = base - timedelta(hours=base.hour % h) + timedelta(seconds=10)
        while cand <= now:
            cand += timedelta(hours=h)
        return cand.astimezone(KST)

    def run(self, once: bool = False) -> None:
        warn = self.store.load()
        if warn:
            log(f"⚠ {warn}")
        if self.stop.active:
            print(f"\n{'=' * 50}\nThe bot is stopped.\n\n{self.stop.reason()}\n{'=' * 50}")
            return

        log(f"■ Start\n{describe(self.cfg)}")
        if not os.path.exists(ENV_PATH):
            log("⚠ No .env found; using defaults in dry-run mode without API keys.\n"
                "     → Copy .env.example to .env and fill in your keys.\n"
                "     → Ensure the filename is .env, not .env.txt.\n"
                f"     → Expected path: {ENV_PATH}")
        if not self.cfg.dry_run:
            log("LIVE MODE: real funds will be traded.")

        noti_err = notify.check(self.cfg)
        if noti_err:
            log(f"⚠ Notifications unavailable: {noti_err}\n"
                "     -> Check both TELEGRAM_ settings in .env to receive failure notifications.")
        else:
            log("  Notifications connected: trades, errors, and stops will be sent to Telegram")
            self.alert(f"▶️ Starting trading bot\n{self.cfg.market} / {self.cfg.timeframe} / "
                       f"{self.cfg.strategy}\n"
                       f"{'Dry run (no orders)' if self.cfg.dry_run else 'LIVE MODE'}")

        while True:
            price: Optional[float] = None
            krw: Optional[float] = None
            try:
                self.cycle()
                self.last_error = ""
                self.fail_streak = 0
                price = self.api.last_price(self.cfg.market)
                krw = self.read_account(price)[0]
            except SystemExit:
                raise
            except ConfigError as e:
                log(f"■ Stopping due to configuration error\n{e}")
                self.alert(f"🛑 Stopped due to configuration error\n{e}")
                raise SystemExit(2)     # Exit code 2 requires operator intervention; the supervisor must not restart.
            except UpbitError as e:
                self.last_error = f"{e} ({fmt(now_kst())})"
                self.fail_streak += 1
                log(f"⚠ {e.full()}")
                self._alert_failure(e.full())
            except Exception as e:
                self.last_error = f"{type(e).__name__}: {e} ({fmt(now_kst())})"
                self.fail_streak += 1
                log(f"⚠ Unexpected error: {type(e).__name__}: {e}\n"
                    f"     → Retrying next cycle. Investigate the traceback if this persists.\n"
                    f"{traceback.format_exc()}")
                self._alert_failure(f"{type(e).__name__}: {e}")

            self.next_wake = self.next_bar_close()
            self.write_status(price, krw)
            self.maybe_heartbeat(price, krw)
            if once:
                log("■ Single check completed; exiting (--once)")
                return
            log(f"  Next check at {fmt(self.next_wake)}")
            last_watch = 0.0
            while now_kst() < self.next_wake:
                if self.stop.active:
                    log("■ Stop file detected; exiting")
                    return
                if time.monotonic() - last_watch >= 60.0:
                    last_watch = time.monotonic()
                    try:
                        self.watch_between_bars()
                    except UpbitError as e:
                        log(f"⚠ Stop-loss check failed: {e}")
                    except Exception as e:
                        log(f"⚠ Unexpected stop-loss check error: {type(e).__name__}: {e}")
                time.sleep(min(30.0, max(1.0, (self.next_wake - now_kst()).total_seconds())))


def main() -> None:
    once = "--once" in sys.argv
    try:
        cfg = load_config()
    except ConfigError as e:
        print(f"\n{e}\n")
        sys.exit(2)                 # Exit code 2 requires operator intervention; do not auto-restart.
    Bot(cfg).run(once=once)


if __name__ == "__main__":
    main()
