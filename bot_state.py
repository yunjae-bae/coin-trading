# -*- coding: utf-8 -*-
from __future__ import annotations

import json
import os
import shutil
from dataclasses import asdict, dataclass, field
from datetime import datetime, timedelta, timezone
from typing import Any, Dict, Optional

KST = timezone(timedelta(hours=9))


def now_kst() -> datetime:
    return datetime.now(KST)


def fmt(dt: Optional[datetime]) -> str:
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
class State:
    position: Position = field(default_factory=Position)
    pending_exit: bool = False
    # Partial fills still need an exit when the original signal has passed.
    last_bar_attempted: str = ""
    day_date: str = ""               # KST date for the daily loss reference.
    day_start_equity: float = 0.0
    last_trade_time: str = ""
    last_error: str = ""
    orphan_notified: str = ""
    heartbeat_date: str = ""         # KST date of the last heartbeat.

    def to_json(self) -> Dict[str, Any]:
        d = asdict(self)
        return d

    @classmethod
    def from_json(cls, d: Dict[str, Any]) -> "State":
        pos = Position(**{k: v for k, v in (d.get("position") or {}).items()
                          if k in Position.__dataclass_fields__})
        rest = {k: v for k, v in d.items() if k in cls.__dataclass_fields__ and k != "position"}
        return cls(position=pos, **rest)


class StateStore:
    def __init__(self, path: str):
        self.path = path
        self.state = State()

    def _read(self, path: str) -> bool:
        with open(path, "r", encoding="utf-8") as f:
            self.state = State.from_json(json.load(f))
        return True

    def load(self) -> str:
        """Recover the backup on read failure; return a warning or an empty string."""
        if not os.path.exists(self.path):
            return ""
        try:
            self._read(self.path)
            return ""
        except (OSError, ValueError, TypeError, json.JSONDecodeError) as e:
            broken = self.path + ".broken"
            try:
                os.replace(self.path, broken)
            except OSError:
                pass
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
        with open(tmp, "r", encoding="utf-8") as f:
            json.load(f)
        if os.path.exists(self.path):
            try:
                shutil.copy2(self.path, self.path + ".bak")
            except OSError:
                pass
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


class RunLock:
    STALE_SEC = 120

    def __init__(self, path: str):
        self.path = path

    def _age(self) -> Optional[float]:
        try:
            with open(self.path, "r", encoding="utf-8") as f:
                t = datetime.fromisoformat(f.read().strip())
            return (now_kst() - t).total_seconds()
        except (OSError, ValueError):
            return None

    def held_by_other(self) -> bool:
        age = self._age()
        return age is not None and 0 <= age < self.STALE_SEC

    def touch(self) -> None:
        try:
            os.makedirs(os.path.dirname(self.path), exist_ok=True)
            with open(self.path, "w", encoding="utf-8") as f:
                f.write(now_kst().isoformat())
        except OSError:
            pass

    def release(self) -> None:
        try:
            os.remove(self.path)
        except OSError:
            pass


class StopFlag:

    def __init__(self, path: str):
        self.path = path

    @property
    def active(self) -> bool:
        return os.path.exists(self.path)

    def reason(self) -> str:
        try:
            with open(self.path, "r", encoding="utf-8") as f:
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
        try:
            os.remove(self.path)
        except OSError:
            pass
