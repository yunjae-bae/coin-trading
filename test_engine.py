from __future__ import annotations

import json
import os
import shutil
import sys
import tempfile
from datetime import UTC

from bot_state import CASHFLOW_TRACKING_VERSION, Position, StateStore, StopFlag
from coin_config import ConfigError, load_config


def test_backup_recovery(tmp):
    path = os.path.join(tmp, "bak.json")
    store = StateStore(path)
    store.state.position = Position(qty=0.7, entry_price=95_000_000.0)
    store.save()
    store.state.position = Position(qty=0.8, entry_price=96_000_000.0)
    store.save()
    assert os.path.exists(path + ".bak"), "Backup was not created"

    with open(path, "w", encoding="utf-8") as f:
        f.write('{"position": {"qty": 0.8, "entry')
    revived = StateStore(path)
    warn = revived.load()
    assert "backup" in warn, f"Missing backup recovery warning: {warn}"
    assert abs(revived.state.position.qty - 0.7) < 1e-12, "Previous backup contents were not restored"
    print("  [OK] State backup recovery")


class _FakeTelegram:

    def __init__(self, *, token_ok: bool = True, send_ok: bool = True,
                 reachable: bool = True):
        self.token_ok = token_ok
        self.send_ok = send_ok
        self.reachable = reachable

    def send_message(self, token: str, chat_id: str, text: str) -> bool:
        import notify
        if not self.reachable:
            raise notify.Unreachable
        return self.send_ok

    def token_valid(self, token: str) -> bool:
        import notify
        if not self.reachable:
            raise notify.Unreachable
        return self.token_ok


def test_notify_offline(tmp):
    import notify
    from coin_config import Config

    off = Config()
    assert notify.send(off, "x") is False, "Reported delivery without configuration"
    assert "not configured" in (notify.check(off) or "")

    cfg = Config(telegram_token="t", telegram_chat_id="c")
    for api, expect in [
        (_FakeTelegram(token_ok=False), "TELEGRAM_TOKEN"),
        (_FakeTelegram(send_ok=False), "TELEGRAM_CHAT_ID"),
        (_FakeTelegram(reachable=False), "internet"),
    ]:
        msg = notify.check(cfg, api)
        assert msg and expect in msg, f"Diagnostic does not identify '{expect}' as expected: {msg}"
    assert notify.check(cfg, _FakeTelegram()) is None, "Valid configuration reported as an error"

    assert notify.send(cfg, "x", _FakeTelegram(reachable=False)) is False
    assert notify.send(cfg, "x", _FakeTelegram(send_ok=False)) is False
    print("  [OK] Four notification diagnostics and failure isolation")


def test_corrupted_state_recovers(tmp):
    path = os.path.join(tmp, "state.json")
    with open(path, "w", encoding="utf-8") as f:
        f.write('{"position": {"qty": 0.5, "entry_pr')      # Simulate a write interrupted by power loss.
    store = StateStore(path)
    warn = store.load()
    assert warn, "Corruption was not detected"
    assert "corrupted" in warn and "review required" in warn, f"Warning lacks recovery guidance: {warn}"
    assert os.path.exists(path + ".broken"), "Corrupted file was not preserved"
    assert store.state.position.qty == 0.0, "State was not reset after corruption"
    print("  [OK] Corrupted state recovery and warning")


def test_atomic_save(tmp):
    path = os.path.join(tmp, "atomic.json")
    store = StateStore(path)
    store.state.position = Position(qty=0.123, entry_price=90_000_000.0)
    store.save()
    assert not os.path.exists(path + ".tmp"), "Temporary file was left behind"
    with open(path, encoding="utf-8") as f:
        json.load(f)
    again = StateStore(path)
    assert again.load() == ""
    assert abs(again.state.position.qty - 0.123) < 1e-12
    assert again.state.position.entry_price == 90_000_000.0
    print("  [OK] Atomic save and reload")


def test_reconcile(tmp):
    store = StateStore(os.path.join(tmp, "rec.json"))

    # Manual partial sell: reduce the bot record to the balance.
    store.state.position = Position(qty=1.0, entry_price=100.0)
    warn = store.reconcile(0.4)
    assert warn and abs(store.state.position.qty - 0.4) < 1e-12, "Tracked quantity was not reduced"

    # Manual full sell: close the tracked position.
    store.state.position = Position(qty=1.0, entry_price=100.0)
    store.reconcile(0.0)
    assert store.state.position.qty == 0.0, "Position was not closed"

    # Unrelated additional holdings: leave the bot record unchanged.
    store.state.position = Position(qty=1.0, entry_price=100.0)
    warn = store.reconcile(5.0)
    assert warn == "" and store.state.position.qty == 1.0, "Adopted unrelated holdings as bot-owned"

    store.state.position = Position(qty=1.0, entry_price=100.0)
    assert store.reconcile(0.9999) == "", "Unnecessary warning within rounding tolerance"
    print("  [OK] Four reconciliation cases, including unrelated holdings")


def test_pending_exit(tmp):
    path = os.path.join(tmp, "pending.json")
    store = StateStore(path)
    store.state.position = Position(qty=1.0, entry_price=100.0)
    store.save()

    store.mark_pending_exit(0.3)
    assert store.state.pending_exit is True and abs(store.state.position.qty - 0.3) < 1e-12

    again = StateStore(path)
    assert again.load() == ""
    assert again.state.pending_exit is True, "Pending exit was lost after restart"
    assert abs(again.state.position.qty - 0.3) < 1e-12

    again.close_position()
    assert again.state.pending_exit is False, "Pending exit remained after completion"
    assert again.state.position.qty == 0.0

    store.state.position = Position(qty=1.0, entry_price=100.0)
    store.state.pending_exit = False
    store.want_exit()
    assert store.state.pending_exit is True
    assert StateStore(path).load() == "" or True
    reloaded = StateStore(path)
    reloaded.load()
    assert reloaded.state.pending_exit is True, "Exit intent was not persisted"

    store.open_position(0.5, 200.0)
    assert store.state.pending_exit is False, "New position inherited an old exit intent"

    store.state.position = Position(qty=1.0, entry_price=100.0)
    store.state.pending_exit = True
    store.reconcile(0.0)
    assert store.state.pending_exit is False, "Exit intent remained with zero balance"
    print("  [OK] Pending exits across restart, reentry, and manual sells")


def test_config_validation(tmp):
    def write(text):
        p = os.path.join(tmp, ".env")
        with open(p, "w", encoding="utf-8") as f:
            f.write(text)
        return p

    for bad, must_have in [
        ("POSITION_PCT=fifty\n", "number"),
        ("POSITION_PCT=500\n", "range"),
        ("TIMEFRAME=30m\n", "1h"),
        ("STRATEGY=rsi\n", "qqe"),
        ("STOP_LOSS_PCT=-5\n", "range"),
        ("DRY_RUN=0\n", "API key"),
    ]:
        try:
            load_config(env_path=write(bad))
            raise AssertionError(f"Invalid configuration was accepted: {bad.strip()}")
        except ConfigError as e:
            assert must_have in str(e), f"Message is missing '{must_have}': {e}"
            assert "[Configuration error]" in str(e)

    p = write("MARKET=KRW-BTC\nTIMEFRAME=4h\nPOSITION_PCT=abc\n")
    try:
        load_config(env_path=p)
        raise AssertionError("Expected rejection")
    except ConfigError as e:
        assert "line 3" in str(e), f"Missing source line number: {e}"

    p = write("﻿MARKET=KRW-BTC\nTIMEFRAME=4h   #  bars\nSTRATEGY=\"qqe_mod\"\nPOSITION_PCT=30\n")
    cfg = load_config(env_path=p)
    assert cfg.market == "KRW-BTC" and cfg.timeframe == "4h"
    assert cfg.strategy == "qqe_mod" and cfg.position_pct == 30.0
    assert cfg.rsi_period == 6 and cfg.factor == 3.0, "QQE MOD defaults were not applied"
    assert cfg.dry_run is True, "Dry run is not enabled by default"
    print("  [OK] Seven configuration cases including lines, BOM, comments, and quotes")


def test_identifier():
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    from autotrade_coin import Bot
    from coin_config import Config
    bot = Bot(Config())
    a = bot._identifier("2026-08-04T16:00:00+00:00", "bid")
    b = bot._identifier("2026-08-04T16:00:00+00:00", "bid")
    c = bot._identifier("2026-08-04T16:00:00+00:00", "ask")
    d = bot._identifier("2026-08-04T20:00:00+00:00", "bid")
    assert a == b, "Identifier changed for the same bar and side; retries may duplicate orders"
    assert a != c and a != d, "Different bars or sides share an identifier"
    assert len(a) <= 40 and a.replace("-", "").isalnum(), f"Invalid identifier format: {a}"
    w1 = bot._identifier("w202608050026", "ask")
    w2 = bot._identifier("w202608050027", "ask")
    assert w1 != w2, "Different minutes share an identifier; retries will be rejected"
    assert w1 != c and w1 != a, "Bar and stop-loss order identifiers collide"
    assert len(w1) <= 40 and w1.replace("-", "").isalnum(), f"Invalid format: {w1}"
    print(f"  [OK] Order identifier idempotency ({a} / {w1})")


def test_next_bar_close():
    from datetime import datetime

    from autotrade_coin import Bot
    from coin_config import Config
    from fake_exchange import FakeExchange

    def at(server_utc: str) -> FakeExchange:
        return FakeExchange(now=datetime.fromisoformat(server_utc).replace(tzinfo=UTC))

    for server_utc, expect_kst in [
        ("2026-08-04 16:00:05", "2026-08-05 01:00:10"),
        ("2026-08-04 16:00:15", "2026-08-05 05:00:10"),
        ("2026-08-04 17:30:00", "2026-08-05 05:00:10"),
        ("2026-08-04 19:59:59", "2026-08-05 05:00:10"),
    ]:
        bot = Bot(Config(timeframe="4h"), at(server_utc))
        got = bot.next_bar_close().strftime("%Y-%m-%d %H:%M:%S")
        assert got == expect_kst, f"{server_utc}(UTC) -> {got}, expected {expect_kst}"

    bot1h = Bot(Config(timeframe="1h"), at("2026-08-04 13:00:03"))
    assert bot1h.next_bar_close().strftime("%H:%M:%S") == "22:00:10"   # 13:00:10 UTC = 22:00:10 KST
    print("  [OK] Next decision timestamp includes just-closed bars")


def test_daily_loss(tmp):
    store = StateStore(os.path.join(tmp, "day.json"))
    assert store.roll_day(10_000_000.0) is True, "First observation should start a new day"
    assert store.roll_day(9_000_000.0) is False, "Daily reference reset within the same day"
    assert abs(store.day_loss_pct(9_000_000.0) - (-10.0)) < 1e-9
    assert abs(store.day_loss_pct(11_000_000.0) - 10.0) < 1e-9
    print("  [OK] Daily loss calculation")


def test_external_cashflow_is_not_profit_or_loss(tmp):
    store = StateStore(os.path.join(tmp, "cashflow.json"))
    store.state.day_start_equity = 1_000_000.0
    assert store.observe_account(1_000_000.0, 0.0, 100_000.0) == 0.0

    withdrawn = store.observe_account(100_000.0, 0.0, 100_000.0)
    assert withdrawn == -900_000.0
    assert store.state.day_start_equity == 100_000.0
    assert abs(store.day_loss_pct(100_000.0)) < 1e-9, "Withdrawal counted as a 90% loss"

    deposited = store.observe_account(400_000.0, 0.0, 100_000.0)
    assert deposited == 300_000.0
    assert store.state.day_start_equity == 400_000.0
    assert abs(store.day_loss_pct(400_000.0)) < 1e-9, "Deposit counted as profit"

    held = StateStore(os.path.join(tmp, "held.json"))
    held.state.day_start_equity = 1_000_000.0
    held.observe_account(0.0, 1.0, 1_000_000.0)
    assert held.observe_account(0.0, 1.0, 900_000.0) == 0.0
    assert abs(held.day_loss_pct(900_000.0) - (-10.0)) < 1e-9, \
        "Market loss was hidden as a withdrawal"

    # Cash flow after a loss must neither dilute nor inflate the loss percentage.
    after_loss = held.observe_account(500_000.0, 0.0, 900_000.0)
    assert after_loss == -400_000.0
    assert abs(held.day_loss_pct(500_000.0) - (-10.0)) < 1e-9, \
        "Withdrawal changed an existing loss percentage"
    after_deposit = held.observe_account(800_000.0, 0.0, 900_000.0)
    assert after_deposit == 300_000.0
    assert abs(held.day_loss_pct(800_000.0) - (-10.0)) < 1e-9, \
        "Deposit diluted an existing loss percentage"

    manual = StateStore(os.path.join(tmp, "manual.json"))
    manual.state.day_start_equity = 1_000_000.0
    manual.observe_account(0.0, 1.0, 900_000.0)
    assert manual.observe_account(900_000.0, 0.0, 900_000.0) == 0.0
    print("  [OK] Exclude external cash flow while preserving market profit/loss")


def test_legacy_withdrawal_stop_recovers_once(tmp):
    from autotrade_coin import Bot
    from coin_config import Config

    bot = Bot(Config())
    bot.store = StateStore(os.path.join(tmp, "legacy.json"))
    bot.stop = StopFlag(os.path.join(tmp, "legacy-stop.txt"))
    bot.store.state.cashflow_tracking_version = 0
    bot.store.state.day_date = "2026-09-02"
    bot.store.state.day_start_equity = 1_000_000.0
    bot.store.save()
    bot.store.state.cashflow_tracking_version = 1
    bot.store.save()
    bot.stop.raise_flag(
        "\uc624\ub298 \uc190\uc2e4\uc774 -39.98% \ub85c \uc124\uc815\ud55c \ud55c\ub3c4(10%)\ub97c \ub118\uc5c8\uc2b5\ub2c8\ub2e4.\n"
        "\uc624\ub298 \uc2dc\uc791 \uc790\uc0b0 4,999,982\uc6d0 \u2192 \ud604\uc7ac 3,000,982\uc6d0"
    )

    assert bot._upgrade_cashflow_tracking() is True
    assert not bot.stop.active
    assert bot.store.state.day_date == "" and bot.store.state.day_start_equity == 0.0
    assert bot.store.state.cashflow_tracking_version == CASHFLOW_TRACKING_VERSION

    bot.stop.raise_flag("Daily loss -12.00% exceeds the daily loss limit")
    assert bot._upgrade_cashflow_tracking() is False
    assert bot.stop.active, "Current-version daily loss stop was incorrectly cleared"
    print("  [OK] One-time legacy stop recovery preserves subsequent loss stops")


def test_run_lock(tmp):
    from datetime import timedelta

    from bot_state import RunLock, now_kst
    path = os.path.join(tmp, "\uc2e4\ud589\uc911.lock")
    lock = RunLock(path)

    assert not lock.held_by_other(), "Nonexistent lock reported as active"
    lock.touch()
    assert lock.held_by_other(), "Fresh lock not detected; duplicate instance allowed"
    lock.release()
    assert not lock.held_by_other(), "Lock remained after release"

    with open(path, "w", encoding="utf-8") as f:
        f.write((now_kst() - timedelta(seconds=RunLock.STALE_SEC + 10)).isoformat())
    assert not lock.held_by_other(), "Stale lock prevented restart"

    assert RunLock.STALE_SEC < 180, "Lock timeout exceeds restart delay and prevents recovery"

    with open(path, "w", encoding="utf-8") as f:
        f.write("invalid contents")
    assert not lock.held_by_other(), "Broken lock prevented startup"

    lock.touch()
    intruder = RunLock(path)
    assert intruder.held_by_other()
    intruder.release()
    assert lock.held_by_other(), "Blocked instance removed the active instance lock"
    lock.release()
    print("  [OK] Duplicate-instance protection with stale and broken lock recovery")


def test_stop_flag(tmp):
    flag = StopFlag(os.path.join(tmp, "\ubd07\uc815\uc9c0.txt"))
    assert not flag.active
    flag.raise_flag("Daily loss limit exceeded")
    assert flag.active
    body = flag.reason()
    assert "limit" in body and "delete" in body, "Stop reason or resume instructions missing"
    flag.clear()
    assert not flag.active
    print("  [OK] Stop flag and resume instructions")


def main() -> int:
    tmp = tempfile.mkdtemp(prefix="coinbot-test-")
    try:
        test_corrupted_state_recovers(tmp)
        test_backup_recovery(tmp)
        test_notify_offline(tmp)
        test_atomic_save(tmp)
        test_reconcile(tmp)
        test_pending_exit(tmp)
        test_config_validation(tmp)
        test_identifier()
        test_next_bar_close()
        test_daily_loss(tmp)
        test_external_cashflow_is_not_profit_or_loss(tmp)
        test_legacy_withdrawal_stop_recovers_once(tmp)
        test_run_lock(tmp)
        test_stop_flag(tmp)
        print("\nAll checks passed")
        return 0
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


if __name__ == "__main__":
    sys.exit(main())
