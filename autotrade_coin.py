from __future__ import annotations

import csv
import math
import os
import sys
import time
import traceback
from datetime import UTC, datetime, timedelta

import pandas as pd

import notify
from bot_state import (
    CASHFLOW_TRACKING_VERSION,
    KST,
    PendingOrder,
    Position,
    RunLock,
    StateStore,
    StopFlag,
    fmt,
    now_kst,
)
from coin_config import BASE_DIR, ENV_PATH, Config, ConfigError, describe, load_config
from coin_kernel import (
    REASON_STOP,
    decide_last_bar,
    resample_ohlcv,
    stop_loss_hit,
    warmup_bars,
)
from upbit_api import Exchange, Upbit, UpbitError

OUT_DIR = os.path.join(BASE_DIR, "out")
LOG_PATH = os.path.join(OUT_DIR, "trading.log")
ORDERS_PATH = os.path.join(OUT_DIR, "orders.csv")
STATUS_PATH = os.path.join(OUT_DIR, "status.txt")
STATE_PATH = os.path.join(OUT_DIR, "state.json")
# Retain legacy stop and lock paths so upgrades cannot bypass an active halt or instance.
STOP_PATH = os.path.join(OUT_DIR, "\ubd07\uc815\uc9c0.txt")
LOCK_PATH = os.path.join(OUT_DIR, "\uc2e4\ud589\uc911.lock")

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
    def __init__(self, cfg: Config, api: Exchange | None = None):
        self.cfg = cfg
        self.api: Exchange = api or Upbit(cfg.access_key, cfg.secret_key)
        runtime = os.path.join(OUT_DIR, "paper") if cfg.dry_run else OUT_DIR
        self.store = StateStore(os.path.join(runtime, os.path.basename(STATE_PATH)))
        self.stop = StopFlag(os.path.join(runtime, os.path.basename(STOP_PATH)))
        self.lock = RunLock(LOCK_PATH)
        self.last_error = ""
        self.last_decision = "-"
        self.fail_streak = 0
        self.next_wake: datetime | None = None

    def alert(self, text: str) -> None:
        try:
            tag = "" if not self.cfg.dry_run else "[dry run] "
            notify.send(self.cfg, tag + text)
        except Exception as e:
            log(f"  (Notification delivery failed: {type(e).__name__})")


    def _ensure_market(self) -> None:
        st = self.store.state
        recorded = st.market or (st.pending_order.market if st.pending_order else "")
        if st.position.is_open and not recorded:
            raise ConfigError("Legacy position has no market. Verify its ownership and set the state market before resuming.")
        if recorded and recorded != self.cfg.market and (st.position.is_open or st.pending_order):
            raise ConfigError(f"Tracked position or order belongs to {recorded}; restore MARKET before resuming.")
        if st.market != self.cfg.market:
            st.market = self.cfg.market
            st.day_date = ""
            st.account_observed = False
            st.last_exit_time = ""
            st.last_bar_attempted = ""
            self.store.save()

    def fetch_bars(self) -> pd.DataFrame:
        need_1h = (warmup_bars(self.cfg) + 5) * self.cfg.bar_hours
        df_1h = self.api.candles_1h(self.cfg.market, need_1h)
        bars = resample_ohlcv(df_1h, self.cfg.timeframe)
        if len(bars) < warmup_bars(self.cfg):
            raise UpbitError(
                f"Insufficient historical bars for a decision ({len(bars)} / {warmup_bars(self.cfg)} required)",
                "Retrying shortly.")
        return bars

    def read_account(self) -> tuple[float, float]:
        self._ensure_market()
        if self.cfg.dry_run:
            return self.store.state.paper_krw, self.store.state.position.qty
        return self.api.balances(self.cfg.market)


    def _identifier(self, bar_iso: str, side: str) -> str:
        return f"qqe-{bar_iso.replace('-', '').replace(':', '')[:13]}-{side}"

    def buy(self, krw: float, price: float | None, bar_iso: str, reason: str,
            bar_close: float = 0.0) -> None:
        self._ensure_market()
        if self.store.state.pending_order is not None:
            self._resolve_order()
            return
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
            if not price or price <= 0:
                return
            qty = order_krw / price
            st = self.store.state
            st.paper_krw -= order_krw * (1 + self.cfg.fee_rate)
            st.position = Position(qty, price, fmt(now_kst()))
            st.pending_exit = False
            st.last_trade_time = fmt(now_kst())
            self.store.record_account(self.store.state.paper_krw, qty)
            self.alert(f"🟢 buy\n{self.cfg.market} {order_krw:,.0f} KRW\n"
                       f"Price {price:,.0f} KRW\nReason: {reason}" if price else
                       f"🟢 buy\n{self.cfg.market} {order_krw:,.0f} KRW\nReason: {reason}")
            log_order({"timestamp": fmt(now_kst()), "side": "buy", "reason": reason,
                       "market": self.cfg.market, "quantity": f"{qty:.8f}", "fill_price": price or "",
                       "amount": f"{order_krw:.0f}", "mode": "dry run", "order_id": ident})
            return

        self._submit_order(ident, "bid", order_krw, reason)

    def _submit_order(self, identifier: str, side: str, amount: float, reason: str) -> None:
        st = self.store.state
        st.order_sequence += 1
        identifier = f"{identifier}-{st.order_sequence}"
        st.pending_order = PendingOrder(identifier, side, self.cfg.market, st.position.qty, reason)
        self.store.save()  # Persist intent before an exchange can accept the order.
        try:
            if side == "bid":
                self.api.buy_market(self.cfg.market, amount, identifier)
            else:
                self.api.sell_market(self.cfg.market, amount, identifier)
        except UpbitError as exc:
            # Definitive rejections have no fills; uncertain responses retain their identifier.
            if exc.code in ("under_min_total_bid", "under_min_total_ask",
                            "insufficient_funds_bid", "insufficient_funds_ask"):
                st.pending_order = None
                self.store.save()
                if side == "ask" and exc.code == "under_min_total_ask":
                    self.store.close_position()
                    return
            raise
        self._resolve_order()

    def _resolve_order(self) -> bool:
        self._ensure_market()
        st = self.store.state
        pending = st.pending_order
        if pending is None:
            return True
        if pending.market != self.cfg.market:
            raise ConfigError("Resolve the pending order before changing MARKET.")
        state, filled, average = self.api.wait_done(pending.identifier)
        if state not in ("wait", "watch", "done", "cancel"):
            log(f"Order {pending.identifier} remains unresolved.")
            return False
        if (not math.isfinite(filled) or not math.isfinite(average)
                or filled < pending.filled_qty or (filled > 0 and average <= 0)):
            raise UpbitError("Order fill data is incomplete; retaining the pending order.")
        cash, _ = self.api.balances(pending.market)
        self._apply_order_fill(pending, filled, average, cash)
        terminal = state in ("done", "cancel")
        if terminal:
            st.pending_order = None
        try:
            self.store.save()
        except OSError:
            st.pending_order = pending
            raise
        if terminal:
            log_order({"timestamp": fmt(now_kst()), "side": "buy" if pending.side == "bid" else "sell",
                       "reason": pending.reason, "market": pending.market,
                       "quantity": f"{filled:.8f}", "fill_price": f"{average:.0f}" if filled else "",
                       "amount": f"{filled * average:.0f}", "mode": "live",
                       "order_id": pending.identifier})
            self.alert(f"Order {pending.identifier}: {state}; filled {filled:.8f} at {average:,.0f} KRW")
        return terminal

    def _apply_order_fill(self, pending: PendingOrder, filled: float, average: float, cash: float) -> None:
        st = self.store.state
        if pending.side == "bid":
            if filled > 0:
                st.position = Position(pending.position_qty + filled, average, fmt(now_kst()))
        else:
            st.position.qty = max(0.0, pending.position_qty - filled)
            if st.position.qty < 1e-8:
                st.position = Position()
            st.pending_exit = st.position.is_open
            if filled > pending.filled_qty:
                st.last_exit_time = self.api.now_utc().isoformat()
        if filled > 0:
            st.last_trade_time = fmt(now_kst())
        st.account_observed = True
        st.observed_krw = cash
        st.observed_position_qty = st.position.qty
        pending.filled_qty = filled

    def sell(self, price: float | None, bar_iso: str, reason: str) -> None:
        self._ensure_market()
        if self.store.state.pending_order is not None:
            self._resolve_order()
            return
        pos = self.store.state.position
        qty = int(pos.qty * 1e8) / 1e8          # Truncate to eight decimals; rounding up could exceed the balance.
        if qty <= 0:
            log("  Skipping sell: no bot-owned position")
            self.store.close_position()
            return
        # Unsellable dust would otherwise trigger retries on every bar.
        if price and qty * price < self.cfg.min_order_krw:
            log(f"  ⚠ Remaining quantity {qty:.8f} is valued at {qty * price:,.0f} KRW, below the minimum order amount "
                f"({self.cfg.min_order_krw:,.0f} KRW); unable to sell.\n"
                f"     → The bot abandons this dust position and waits for the next signal. "
                f"You can manage it directly in the Upbit app.")
            self.store.close_position()
            return
        ident = self._identifier(bar_iso, "ask")
        if self.cfg.dry_run:
            log(f"  [dry run] sell {qty:.8f} (approximate price {price:,.0f} KRW)" if price
                else f"  [dry run] sell {qty:.8f}")
            if not price or price <= 0:
                return
            pnl = ((price / pos.entry_price - 1) * 100) if pos.entry_price else None
            st = self.store.state
            st.paper_krw += qty * price * (1 - self.cfg.fee_rate)
            st.position = Position()
            st.pending_exit = False
            st.last_trade_time = fmt(now_kst())
            st.last_exit_time = self.api.now_utc().isoformat()
            self.store.record_account(self.store.state.paper_krw, 0.0)
            self.alert(f"🔴 sell\n{self.cfg.market} {qty:.8f}\n"
                       + (f"Price {price:,.0f} KRW ({pnl:+.2f}%)\n" if pnl is not None else "")
                       + f"Reason: {reason}")
            log_order({"timestamp": fmt(now_kst()), "side": "sell", "reason": reason,
                       "market": self.cfg.market, "quantity": f"{qty:.8f}", "fill_price": price or "",
                       "amount": f"{qty * (price or 0):.0f}", "mode": "dry run", "order_id": ident})
            return

        self._submit_order(ident, "ask", qty, reason)


    def _notify_orphan(self, price: float | None, coin_qty: float) -> None:
        cfg = self.cfg
        today = now_kst().strftime("%Y-%m-%d")
        if self.store.state.position.is_open or not price:
            return
        if coin_qty * price < cfg.min_order_krw or self.store.state.orphan_notified == today:
            return
        self.store.state.orphan_notified = today
        self.store.save()
        msg = (f"Account contains {coin_qty:.8f} {cfg.market.split('-')[1]}"
               f"({coin_qty * price:,.0f} KRW) not tracked by the bot.\n"
               f"Manually acquired holdings are left untouched.\n"
               f"Investigate if these are bot-owned holdings with missing state.")
        log(f"  ⚠ {msg}")
        self.alert(f"⚠️ Untracked holdings detected\n{msg}")

    def _halt_on_daily_loss(self, equity: float, mark: float, krw: float) -> None:
        cfg = self.cfg
        if cfg.daily_loss_limit_pct <= 0:
            return
        loss = self.store.day_loss_pct(equity)
        if loss > -cfg.daily_loss_limit_pct:
            return
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

    def _signal_age_min(self, bars: pd.DataFrame) -> float:
        bar_end = bars.index[-1].to_pydatetime() + timedelta(hours=self.cfg.bar_hours)
        return (self.api.now_utc() - bar_end).total_seconds() / 60.0

    def _too_old_to_buy(self, age_min: float) -> bool:
        limit = self.cfg.signal_max_age_min
        if limit <= 0 or age_min <= limit:
            return False
        log(f"  Buy deferred: signal bar closed {age_min:.0f} minutes ago "
            f"(allowed {limit:g} minutes). Stale entry signals are ignored.")
        return True

    def _decide(self, bars: pd.DataFrame) -> tuple[str | None, str]:
        st = self.store.state
        if st.pending_exit and st.position.is_open:
            return "SELL", "Close remaining position"
        action, reason = decide_last_bar(bars, self.cfg, st.position.is_open, st.position.entry_price)
        if action == "BUY" and st.last_exit_time and self.cfg.cooldown_bars > 0:
            exit_bar = pd.Timestamp(st.last_exit_time).tz_convert("UTC").floor(f"{self.cfg.bar_hours}h")
            if bars.index[-1] <= exit_bar + timedelta(hours=self.cfg.bar_hours * self.cfg.cooldown_bars):
                return None, "Entry cooldown"
        return action, reason

    def cycle(self) -> None:
        if not self._resolve_order():
            return
        cfg = self.cfg
        bars = self.fetch_bars()
        price = self.api.last_price(cfg.market)
        krw, coin_qty = self.read_account()

        warn = self.store.reconcile(coin_qty)
        if warn:
            log(f"  ⚠ {warn}")
            self.alert(f"⚠️ Account balance differs from bot records\n{warn}")
        self._notify_orphan(price, coin_qty)

        pos = self.store.state.position
        close = float(bars["close"].iloc[-1])
        # Use the latest close if the price request fails; entry price would hide losses.
        mark = price or close
        equity = krw + pos.qty * mark
        new_day = self.store.roll_day(equity)
        if new_day:
            log(f"  Starting daily reference equity: {equity:,.0f} KRW")
        external_flow = self.store.observe_account(
            krw, pos.qty, mark, adjust_day_base=not new_day)
        if external_flow:
            direction = "deposit" if external_flow > 0 else "withdrawal"
            log(f"  External cash flow: {direction} of approximately {abs(external_flow):,.0f} KRW detected; "
                f"excluded from daily profit/loss")
        self._halt_on_daily_loss(equity, mark, krw)

        action, reason = self._decide(bars)
        self.last_decision = f"{reason} ({fmt(bars.index[-1].to_pydatetime())}  bars)"

        age_min = self._signal_age_min(bars)
        if age_min > cfg.bar_hours * 60 + 30:
            log(f"  ⚠ Market data is {age_min / 60:.1f} hours old. Upbit may be under maintenance or "
                f"market data has stopped updating. Skipping this decision.")
            return
        if action == "BUY" and self._too_old_to_buy(age_min):
            action, reason = None, f"Stale signal({age_min:.0f} minutes ago)"
            self.last_decision = f"{reason} — Buy deferred"

        held = f"holding {pos.qty:.8f} (Entry price {pos.entry_price:,.0f} KRW)" if pos.is_open else "No position"
        log(f"  Bar {fmt(bars.index[-1].to_pydatetime())} Close {close:,.0f} KRW | {held} | "
            f"KRW balance {krw:,.0f} KRW → {action or 'wait'} ({reason})")
        if not action:
            return

        # Record the attempt before submitting: a lost response followed by restart
        # must not create another buy for the same bar.
        bar_iso = bars.index[-1].isoformat()
        if self.store.state.last_bar_attempted == bar_iso:
            log(f"  Skipping: bar ({bar_iso}) was already processed")
            return
        self.store.state.last_bar_attempted = bar_iso
        self.store.save()

        if action == "BUY":
            self.buy(krw, price, bar_iso, reason, close)
        else:
            self.store.want_exit()
            self.sell(price, bar_iso, reason)


    def watch_between_bars(self) -> None:
        if not self._resolve_order():
            return
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


    def write_status(self, price: float | None, krw: float | None) -> None:
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

    def maybe_heartbeat(self, price: float | None, krw: float | None) -> None:
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
        now = self.api.now_utc().astimezone(UTC)
        h = self.cfg.bar_hours
        base = now.replace(minute=0, second=0, microsecond=0)
        cand = base - timedelta(hours=base.hour % h) + timedelta(seconds=10)
        while cand <= now:
            cand += timedelta(hours=h)
        return cand.astimezone(KST)

    def _blocked(self) -> bool:
        if self.stop.active:
            print(f"\n{'=' * 50}\nThe bot is stopped.\n\n{self.stop.reason()}\n{'=' * 50}")
            return True
        return False

    def _upgrade_cashflow_tracking(self) -> None:
        st = self.store.state
        if st.cashflow_tracking_version >= CASHFLOW_TRACKING_VERSION:
            return
        st.cashflow_tracking_version = CASHFLOW_TRACKING_VERSION
        st.account_observed = False
        st.observed_krw = 0.0
        st.observed_position_qty = 0.0
        self.store.save()
        if self.stop.active:
            log("Legacy halt retained.")

    def _greet(self) -> None:
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
                "     → Check both TELEGRAM_ settings to receive failure notifications.")
            return
        log("  Notifications connected: trades, errors, and stops will be sent to Telegram")
        self.alert(f"▶️ Starting trading bot\n{self.cfg.market} / {self.cfg.timeframe} / "
                   f"{self.cfg.strategy}\n"
                   f"{'Dry run (no orders)' if self.cfg.dry_run else 'LIVE MODE'}")

    def _step(self) -> tuple[float | None, float | None]:
        """Return price and KRW, or (None, None) on failure. Exit on configuration errors."""
        try:
            self.cycle()
            if self.store.state.pending_order is not None:
                return None, None
            self.last_error = ""
            self.fail_streak = 0
            price = self.api.last_price(self.cfg.market)
            krw, _ = self.read_account()
            # Refresh observations after successful orders so later price changes
            # are not mistaken for deposits or withdrawals.
            self.store.record_account(krw, self.store.state.position.qty)
            return price, krw
        except SystemExit:
            raise
        except ConfigError as e:
            log(f"■ Stopping due to configuration error\n{e}")
            self.alert(f"🛑 Stopped due to configuration error\n{e}")
            # Exit code 2 requires operator intervention; the supervisor must not restart.
            raise SystemExit(2) from e
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
        return None, None

    def _watch_guarded(self) -> None:
        try:
            self.watch_between_bars()
        except UpbitError as e:
            log(f"⚠ Stop-loss check failed: {e}")
        except Exception as e:
            log(f"⚠ Unexpected stop-loss check error: {type(e).__name__}: {e}")

    def _wait_until(self, until: datetime) -> bool:
        """Return False when a stop flag requests shutdown."""
        last_watch = 0.0
        while now_kst() < until:
            if self.stop.active:
                log("■ Stop file detected; exiting")
                return False
            if time.monotonic() - last_watch >= 60.0:
                last_watch = time.monotonic()
                self._watch_guarded()
            time.sleep(min(30.0, max(1.0, (until - now_kst()).total_seconds())))
        return True

    def run(self, once: bool = False) -> None:
        if not self.lock.acquire():
            log("Another bot instance holds the runtime lock; exiting.")
            return
        warn = self.store.load()
        if warn:
            log(f"⚠ {warn}")
        self._ensure_market()
        self._upgrade_cashflow_tracking()
        if self._blocked():
            return
        self._greet()

        while True:
            price, krw = self._step()
            self.next_wake = self.next_bar_close()
            self.write_status(price, krw)
            self.maybe_heartbeat(price, krw)
            if once:
                log("■ Single check completed; exiting (--once)")
                return
            log(f"  Next check at {fmt(self.next_wake)}")
            if not self._wait_until(self.next_wake):
                self.lock.release()
                return


def main() -> None:
    once = "--once" in sys.argv
    try:
        cfg = load_config()
    except ConfigError as e:
        print(f"\n{e}\n")
        sys.exit(2)                 # Exit code 2 requires operator intervention; do not auto-restart.
    bot = Bot(cfg)
    try:
        bot.run(once=once)
    except ConfigError as e:
        print(f"\n{e}\n")
        raise SystemExit(2) from e
    finally:
        bot.lock.release()


if __name__ == "__main__":
    main()
