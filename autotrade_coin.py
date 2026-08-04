# -*- coding: utf-8 -*-
from __future__ import annotations

import base64
import csv
import hashlib
import hmac
import json
import os
import time
import uuid
from datetime import datetime, timedelta, timezone
from typing import Any, Dict, List, Optional, Tuple

import pandas as pd
import requests

from coin_kernel import FEE_RATE, MARKET, MIN_ORDER_KRW, WARMUP_HOURS, decide_last_bar

BASE_URL = "https://api.upbit.com"
BASE_DIR = os.path.abspath(os.path.dirname(__file__))
OUT_DIR = os.path.join(BASE_DIR, "out")

try:
    from dotenv import load_dotenv
except ImportError:
    load_dotenv = None

_env_file = os.getenv("ENV_FILE", "")
if load_dotenv is not None:
    for fp in [_env_file, "coin.cfg", ".env"]:
        if fp and os.path.exists(fp):
            load_dotenv(fp, override=True)
            print(f"[CONFIG] loaded {fp}")
            break

ACCESS_KEY = os.getenv("UPBIT_ACCESS_KEY", "").strip()
SECRET_KEY = os.getenv("UPBIT_SECRET_KEY", "").strip()
DRY_RUN = (os.getenv("DRY_RUN", "1").strip() != "0")
FETCH_COUNT = 60

ORDERS_HEADER = ["ts", "action", "market", "ord_type", "krw_amount", "volume", "price_ref",
                 "uuid", "state", "dry_run", "detail"]


def _log(line: str) -> None:
    os.makedirs(OUT_DIR, exist_ok=True)
    stamp = datetime.now(timezone.utc).isoformat()
    with open(os.path.join(OUT_DIR, "run.log"), "a", encoding="utf-8", newline="\n") as f:
        f.write(f"{stamp} {line}\n")
    print(f"{stamp} {line}")


def _log_order(row: Dict[str, Any]) -> None:
    os.makedirs(OUT_DIR, exist_ok=True)
    p = os.path.join(OUT_DIR, "orders.csv")
    new = not (os.path.exists(p) and os.path.getsize(p) > 0)
    with open(p, "a", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=ORDERS_HEADER)
        if new:
            w.writeheader()
        w.writerow({k: row.get(k, "") for k in ORDERS_HEADER})


def _b64url(data: bytes) -> str:
    return base64.urlsafe_b64encode(data).rstrip(b"=").decode()


def _jwt(params: Optional[Dict[str, Any]] = None) -> str:
    payload: Dict[str, Any] = {"access_key": ACCESS_KEY, "nonce": str(uuid.uuid4())}
    if params:
        qs = "&".join(f"{k}={v}" for k, v in params.items())
        payload["query_hash"] = hashlib.sha512(qs.encode()).hexdigest()
        payload["query_hash_alg"] = "SHA512"
    header = _b64url(json.dumps({"alg": "HS512", "typ": "JWT"}, separators=(",", ":")).encode())
    body = _b64url(json.dumps(payload, separators=(",", ":")).encode())
    sig = _b64url(hmac.new(SECRET_KEY.encode(), f"{header}.{body}".encode(), hashlib.sha512).digest())
    return f"{header}.{body}.{sig}"


def _private(method: str, path: str, params: Optional[Dict[str, Any]] = None) -> Any:
    headers = {"Authorization": f"Bearer {_jwt(params)}"}
    url = BASE_URL + path
    if method == "GET":
        r = requests.get(url, params=params, headers=headers, timeout=10)
    else:
        headers["Content-Type"] = "application/json; charset=utf-8"
        r = requests.post(url, json=params, headers=headers, timeout=10)
    if r.status_code >= 400:
        raise RuntimeError(f"{method} {path} -> {r.status_code}: {r.text[:300]}")
    return r.json()


def fetch_1h_closed(count: int = FETCH_COUNT) -> pd.DataFrame:
    r = requests.get(f"{BASE_URL}/v1/candles/minutes/60",
                     params={"market": MARKET, "count": count}, timeout=10)
    r.raise_for_status()
    df = pd.DataFrame(r.json())
    df["ts"] = pd.to_datetime(df["candle_date_time_utc"], errors="coerce").dt.tz_localize(timezone.utc)
    df = df.rename(columns={"opening_price": "open", "high_price": "high", "low_price": "low",
                            "trade_price": "close", "candle_acc_trade_volume": "volume"})
    df = df.dropna(subset=["ts"]).set_index("ts").sort_index()
    now = pd.Timestamp(datetime.now(timezone.utc))
    df = df[df.index + pd.Timedelta(hours=1) <= now]
    return df[["open", "high", "low", "close", "volume"]].astype(float)


def fetch_last_price() -> Optional[float]:
    try:
        r = requests.get(f"{BASE_URL}/v1/ticker", params={"markets": MARKET}, timeout=10)
        r.raise_for_status()
        return float(r.json()[0]["trade_price"])
    except (requests.RequestException, ValueError, KeyError, IndexError):
        return None


def fetch_balances() -> Tuple[float, float]:
    rows: List[Dict[str, Any]] = _private("GET", "/v1/accounts")
    krw = btc = 0.0
    for r in rows:
        if r.get("currency") == "KRW":
            krw = float(r.get("balance") or 0.0)
        elif r.get("currency") == "BTC" and str(r.get("unit_currency")) == "KRW":
            btc = float(r.get("balance") or 0.0)
    return krw, btc


def is_long(btc: float, last_price: Optional[float]) -> bool:
    if btc <= 0:
        return False
    px = last_price or 0.0
    return btc * px >= MIN_ORDER_KRW


def _wait_done(order_uuid: str, timeout_sec: float = 30.0) -> str:
    deadline = time.monotonic() + timeout_sec
    state = "unknown"
    while time.monotonic() < deadline:
        try:
            j = _private("GET", "/v1/order", {"uuid": order_uuid})
            state = str(j.get("state") or "")
            if state in ("done", "cancel"):
                return state
        except RuntimeError as e:
            _log(f"[WARN] order poll failed: {e}")
        time.sleep(1.5)
    return state


def market_buy_all(krw: float, price_ref: Optional[float]) -> None:
    order_krw = math_floor_krw(krw / (1.0 + FEE_RATE))
    if order_krw < MIN_ORDER_KRW:
        _log(f"[SKIP] BUY: available KRW {order_krw:,.0f} < minimum {MIN_ORDER_KRW:,.0f}")
        return
    params = {"market": MARKET, "side": "bid", "ord_type": "price",
              "price": str(order_krw), "identifier": f"coinbot-{uuid.uuid4().hex}"}
    if DRY_RUN:
        _log(f"[DRY_RUN] BUY {MARKET} {order_krw:,.0f} KRW @~{price_ref}")
        _log_order({"ts": datetime.now(timezone.utc).isoformat(), "action": "BUY", "market": MARKET,
                    "ord_type": "price", "krw_amount": order_krw, "price_ref": price_ref,
                    "dry_run": 1, "state": "DRY_RUN"})
        return
    j = _private("POST", "/v1/orders", params)
    state = _wait_done(str(j.get("uuid")))
    _log(f"[ORDER] BUY {order_krw:,.0f} KRW uuid={j.get('uuid')} state={state}")
    _log_order({"ts": datetime.now(timezone.utc).isoformat(), "action": "BUY", "market": MARKET,
                "ord_type": "price", "krw_amount": order_krw, "price_ref": price_ref,
                "uuid": j.get("uuid"), "state": state, "dry_run": 0})


def market_sell_all(btc: float, price_ref: Optional[float]) -> None:
    volume = math_floor8(btc)
    if volume <= 0 or (price_ref or 0.0) * volume < MIN_ORDER_KRW:
        _log(f"[SKIP] SELL: quantity {volume} below the minimum order value")
        return
    params = {"market": MARKET, "side": "ask", "ord_type": "market",
              "volume": f"{volume:.8f}", "identifier": f"coinbot-{uuid.uuid4().hex}"}
    if DRY_RUN:
        _log(f"[DRY_RUN] SELL {MARKET} {volume:.8f} BTC @~{price_ref}")
        _log_order({"ts": datetime.now(timezone.utc).isoformat(), "action": "SELL", "market": MARKET,
                    "ord_type": "market", "volume": f"{volume:.8f}", "price_ref": price_ref,
                    "dry_run": 1, "state": "DRY_RUN"})
        return
    j = _private("POST", "/v1/orders", params)
    state = _wait_done(str(j.get("uuid")))
    _log(f"[ORDER] SELL {volume:.8f} BTC uuid={j.get('uuid')} state={state}")
    _log_order({"ts": datetime.now(timezone.utc).isoformat(), "action": "SELL", "market": MARKET,
                "ord_type": "market", "volume": f"{volume:.8f}", "price_ref": price_ref,
                "uuid": j.get("uuid"), "state": state, "dry_run": 0})


def math_floor_krw(x: float) -> float:
    return float(int(x))


def math_floor8(x: float) -> float:
    return int(x * 1e8) / 1e8


_STATE_PATH = os.path.join(OUT_DIR, "state.json")


def _load_last_acted_bar() -> str:
    try:
        with open(_STATE_PATH, "r", encoding="utf-8") as f:
            return str(json.load(f).get("last_acted_bar") or "")
    except (OSError, ValueError, json.JSONDecodeError):
        return ""


def _save_last_acted_bar(bar_iso: str) -> None:
    try:
        os.makedirs(OUT_DIR, exist_ok=True)
        with open(_STATE_PATH, "w", encoding="utf-8") as f:
            json.dump({"last_acted_bar": bar_iso}, f)
    except OSError as e:
        _log(f"[WARN] state save failed: {e}")


def run_cycle() -> None:
    df = fetch_1h_closed()
    if len(df) < WARMUP_HOURS:
        _log(f"[WAIT] completed bars {len(df)} < warmup {WARMUP_HOURS}")
        return
    last_px = fetch_last_price()
    if DRY_RUN and not (ACCESS_KEY and SECRET_KEY):
        krw, btc = 10_000_000.0, 0.0
    else:
        krw, btc = fetch_balances()
    long_now = is_long(btc, last_px)
    action = decide_last_bar(df, long_now)
    last_bar = df.index[-1].isoformat()
    if action and last_bar == _load_last_acted_bar():
        _log(f"[SKIP] bar={last_bar} already processed this bar; suppressing restart replay")
        action = None
    _log(f"[CYCLE] bar={last_bar} long={long_now} krw={krw:,.0f} btc={btc:.8f} -> {action or 'HOLD'}")
    if action == "BUY":
        market_buy_all(krw, last_px)
        _save_last_acted_bar(last_bar)
    elif action == "SELL":
        market_sell_all(btc, last_px)
        _save_last_acted_bar(last_bar)


def main() -> None:
    if not DRY_RUN and not (ACCESS_KEY and SECRET_KEY):
        raise SystemExit("Live mode (DRY_RUN=0) requires UPBIT_ACCESS_KEY and UPBIT_SECRET_KEY.")
    _log(f"[BOOT] coin engine — market={MARKET} dry_run={DRY_RUN} keys={'set' if ACCESS_KEY else 'none'}")
    while True:
        try:
            run_cycle()
        except Exception as e:
            _log(f"[ERROR] cycle failed: {type(e).__name__}: {e}")
        now = datetime.now(timezone.utc)
        next_wake = (now.replace(minute=0, second=0, microsecond=0)
                     + timedelta(hours=1, seconds=10))
        _log(f"[SLEEP] until {next_wake.isoformat()}")
        while datetime.now(timezone.utc) < next_wake:
            time.sleep(min(30.0, max(0.5, (next_wake - datetime.now(timezone.utc)).total_seconds())))


if __name__ == "__main__":
    main()
