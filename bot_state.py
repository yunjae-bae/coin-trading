from __future__ import annotations

import contextlib
import json
import os
import shutil
from dataclasses import asdict, dataclass, field
from datetime import datetime, timedelta, timezone
from typing import Any, TextIO

KST = timezone(timedelta(hours=9))
CASHFLOW_TRACKING_VERSION = 2


def now_kst() -> datetime:
    return datetime.now(KST)


def fmt(dt: datetime | None) -> str:
    return dt.astimezone(KST).strftime("%Y-%m-%d %H:%M:%S") if dt else "-"


@dataclass
class Position:
    qty: float = 0.0            # Quantity acquired and tracked by the bot.
    entry_price: float = 0.0
    entry_time: str = ""

    @property
    def is_open(self) -> bool:
        return self.qty > 0


@dataclass
class PendingOrder:
    identifier: str
    side: str
    market: str
    position_qty: float
    reason: str
    filled_qty: float = 0.0


@dataclass
class State:
    market: str = ""
    position: Position = field(default_factory=Position)
    # Partial fills still need an exit when the original signal has passed.
    pending_exit: bool = False
    last_bar_attempted: str = ""
    day_date: str = ""               # KST date for the daily loss reference.
    day_start_equity: float = 0.0
    last_trade_time: str = ""
    last_error: str = ""
    orphan_notified: str = ""
    heartbeat_date: str = ""         # KST date of the last heartbeat.
    cashflow_tracking_version: int = CASHFLOW_TRACKING_VERSION
    account_observed: bool = False
    observed_krw: float = 0.0
    observed_position_qty: float = 0.0
    pending_order: PendingOrder | None = None
    paper_krw: float = 10_000_000.0
    order_sequence: int = 0
    last_exit_time: str = ""

    def to_json(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_json(cls, d: dict[str, Any]) -> State:
        pos = Position(**{k: v for k, v in (d.get("position") or {}).items()
                          if k in Position.__dataclass_fields__})
        rest = {k: v for k, v in d.items()
                if k in cls.__dataclass_fields__ and k not in ("position", "pending_order")}
        pending = PendingOrder(**d["pending_order"]) if d.get("pending_order") else None
        # A missing version field denotes state predating cash-flow tracking.
        rest["cashflow_tracking_version"] = int(d.get("cashflow_tracking_version") or 0)
        return cls(position=pos, pending_order=pending, **rest)


class StateStore:
    def __init__(self, path: str):
        self.path = path
        self.state = State()

    def _read(self, path: str) -> None:
        with open(path, encoding="utf-8") as f:
            self.state = State.from_json(json.load(f))

    def load(self) -> str:
        """Recover the backup on read failure; return a warning or an empty string."""
        if not os.path.exists(self.path):
            return ""
        try:
            self._read(self.path)
            return ""
        except (OSError, ValueError, TypeError, json.JSONDecodeError) as e:
            broken = self.path + ".broken"
            with contextlib.suppress(OSError):
                os.replace(self.path, broken)
            bak = self.path + ".bak"
            if os.path.exists(bak):
                try:
                    self._read(bak)
                    self.save()
                    return (f"State file corrupted; restored from backup ({type(e).__name__}). "
                            f"Verify the restored quantity against your actual holdings.")
                except (OSError, ValueError, TypeError, json.JSONDecodeError):
                    pass
            self.state = State()
            return (f"State file corrupted; starting with empty state ({type(e).__name__}). "
                    f"The corrupted file was moved to {os.path.basename(broken)} . "
                    f"Existing holdings are now untracked; review required.")

    def save(self) -> None:
        """Atomically replace state and retain the previous snapshot as a backup."""
        os.makedirs(os.path.dirname(self.path), exist_ok=True)
        tmp = self.path + ".tmp"
        with open(tmp, "w", encoding="utf-8") as f:
            json.dump(self.state.to_json(), f, ensure_ascii=False, indent=2)
            f.flush()
            os.fsync(f.fileno())
        with open(tmp, encoding="utf-8") as f:
            json.load(f)
        if os.path.exists(self.path):
            with contextlib.suppress(OSError):
                shutil.copy2(self.path, self.path + ".bak")
        os.replace(tmp, self.path)


    def open_position(self, qty: float, price: float) -> None:
        self.state.position = Position(qty=qty, entry_price=price, entry_time=fmt(now_kst()))
        self.state.pending_exit = False      # An inherited exit intent would immediately sell the new position.
        self.state.last_trade_time = fmt(now_kst())
        self.save()

    def want_exit(self) -> None:
        if not self.state.pending_exit:
            self.state.pending_exit = True
            self.save()

    def close_position(self) -> None:
        self.state.position = Position()
        self.state.pending_exit = False
        self.state.last_trade_time = fmt(now_kst())
        self.save()

    def mark_pending_exit(self, remain_qty: float) -> None:
        self.state.position.qty = remain_qty
        self.state.pending_exit = True
        self.save()

    def reconcile(self, exchange_qty: float) -> str:
        """Reduce tracked quantity to the exchange balance; leave excess holdings untracked."""
        pos = self.state.position
        if pos.qty <= 0:
            return ""
        if exchange_qty < pos.qty * 0.999:
            msg = (f"Bot record({pos.qty:.8f}) exceeds exchange balance({exchange_qty:.8f}). "
                   f"Adjusting the record to match the balance; check for manual sells.")
            if exchange_qty <= 0:
                self.state.position = Position()
                self.state.pending_exit = False
            else:
                pos.qty = exchange_qty
            self.save()
            return msg
        return ""


    def roll_day(self, equity: float) -> bool:
        """Reset the equity reference on a new KST date; return whether it changed."""
        today = now_kst().strftime("%Y-%m-%d")
        if self.state.day_date != today:
            self.state.day_date = today
            self.state.day_start_equity = equity
            self.save()
            return True
        return False

    def day_loss_pct(self, equity: float) -> float:
        base = self.state.day_start_equity
        if base <= 0:
            return 0.0
        return (equity / base - 1.0) * 100.0

    def observe_account(self, krw: float, position_qty: float, mark: float,
                        *, adjust_day_base: bool = True) -> float:
        """Return external cash flow above tolerance; adjust the daily equity reference if enabled."""
        st = self.state
        if not st.account_observed:
            self.record_account(krw, position_qty)
            return 0.0

        expected = st.observed_krw + st.observed_position_qty * mark
        current = krw + position_qty * mark
        external_flow = current - expected
        tolerance = max(5_000.0, abs(expected) * 0.005)
        adjusted = external_flow if adjust_day_base and abs(external_flow) > tolerance else 0.0
        if adjusted:
            # Adjust the reference proportionally to preserve the existing return percentage.
            if expected > 0 and st.day_start_equity > 0:
                st.day_start_equity = max(0.0, st.day_start_equity * current / expected)
            else:
                st.day_start_equity = max(0.0, current)
        st.observed_krw = krw
        st.observed_position_qty = position_qty
        self.save()
        return adjusted

    def record_account(self, krw: float, position_qty: float) -> None:
        st = self.state
        st.account_observed = True
        st.observed_krw = krw
        st.observed_position_qty = position_qty
        self.save()


class RunLock:
    STALE_SEC = 120

    def __init__(self, path: str):
        self.path = path
        self._file: TextIO | None = None

    def acquire(self) -> bool:
        if self._file is not None:
            return True
        os.makedirs(os.path.dirname(self.path), exist_ok=True)
        # The file must remain open after acquire returns to hold the process lock.
        handle = open(self.path, "a+", encoding="utf-8")  # noqa: SIM115
        try:
            self._lock(handle)
        except OSError:
            handle.close()
            return False
        self._file = handle
        # An older bot only writes timestamps; respect its recent heartbeat.
        handle.seek(0)
        previous = handle.read().strip()
        try:
            age = (now_kst() - datetime.fromisoformat(previous)).total_seconds()
        except (ValueError, TypeError):
            age = self.STALE_SEC
        if age < self.STALE_SEC:
            self.release()
            return False
        try:
            handle.seek(0)
            handle.truncate()
            handle.write(f"pid:{os.getpid()}\n")
            handle.flush()
        except OSError:
            self.release()
            raise
        return True

    @staticmethod
    def _lock(handle: TextIO) -> None:
        handle.seek(0)
        if os.name == "nt":
            import msvcrt
            msvcrt.locking(handle.fileno(), msvcrt.LK_NBLCK, 1)
        else:
            import fcntl
            fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)

    def release(self) -> None:
        if self._file is None:
            return
        # Keep the inode: unlinking a locked file permits a second lock file.
        self._file.close()
        self._file = None


class StopFlag:

    def __init__(self, path: str):
        self.path = path

    @property
    def active(self) -> bool:
        return os.path.exists(self.path)

    def reason(self) -> str:
        try:
            with open(self.path, encoding="utf-8") as f:
                return f.read()
        except OSError:
            return "(Unable to read stop reason)"

    def raise_flag(self, why: str) -> None:
        os.makedirs(os.path.dirname(self.path), exist_ok=True)
        with open(self.path, "w", encoding="utf-8") as f:
            f.write(f"The bot has stopped — {fmt(now_kst())}\n\n"
                    f"{why}\n\n"
                    f"To resume:\n"
                    f"  1. Review your account in Upbit.\n"
                    f"  2. This file({os.path.basename(self.path)}): delete it.\n"
                    f"  3. Restart the program.\n")

    def clear(self) -> None:
        with contextlib.suppress(OSError):
            os.remove(self.path)
