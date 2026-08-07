from __future__ import annotations

import base64
import contextlib
import hashlib
import hmac
import json
import time
import uuid
from datetime import UTC, datetime
from typing import Any, Protocol
from urllib.parse import urlencode

import pandas as pd
import requests

BASE_URL = "https://api.upbit.com"
CANDLE_PAGE = 200
PAGE_SLEEP = 0.15          # Pace requests within the candle-group request limit.


class HttpResponse(Protocol):

    status_code: int

    @property
    def text(self) -> str:
        ...

    def json(self) -> Any:
        ...


class Exchange(Protocol):

    def now_utc(self) -> datetime:
        """Exchange server time."""
        ...

    def candles_1h(self, market: str, count: int) -> pd.DataFrame:
        """Completed hourly bars indexed by UTC opening timestamps."""
        ...

    def last_price(self, market: str) -> float | None:
        """Return None if unavailable."""
        ...

    def balances(self, market: str) -> tuple[float, float]:
        """Available KRW and total coin quantity, including locked coins."""
        ...

    def buy_market(self, market: str, krw: float, identifier: str) -> str:
        """Buy by KRW amount; return the order ID or an empty string if unknown."""
        ...

    def sell_market(self, market: str, qty: float, identifier: str) -> str:
        """Sell by coin quantity; return the order ID or an empty string if unknown."""
        ...

    def wait_done(self, identifier: str) -> tuple[str, float, float]:
        """Return status, filled quantity and average price, including partial fills at timeout."""
        ...


class UpbitError(Exception):

    def __init__(self, message: str, hint: str = "", code: str = "", status: int = 0):
        super().__init__(message)
        self.hint = hint
        self.code = code
        self.status = status

    def full(self) -> str:
        return f"{self}" + (f"\n     → {self.hint}" if self.hint else "")


_ERRORS: dict[str, tuple[str, str]] = {
    "no_authorization_ip": ("Request from an unregistered IP (your public IP may have changed)",
                            "Register your current IP in Upbit Open API settings."),
    "expired_access_key": ("API key expired; issue a new key",
                           "Issue a new key in Upbit and update both key fields in .env."),
    "jwt_verification": ("API key authentication failed",
                         "Check UPBIT_ACCESS_KEY / UPBIT_SECRET_KEY in .env. "
                         "Remove accidental leading or trailing whitespace."),
    "invalid_access_key": ("Invalid API access key",
                           "Check UPBIT_ACCESS_KEY in .env. "
                           "Update .env after replacing your API keys."),
    "invalid_query_payload": ("Invalid API request signature",
                              "Check the key values in .env."),
    "no_authorization_token": ("Missing authentication credentials",
                               "Ensure both API key fields are set in .env."),
    "out_of_scope": ("API key lacks the required permissions",
                     "Enable both 'asset inquiry' and 'orders' permissions for the API key."),
    "insufficient_funds_bid": ("Insufficient KRW balance to buy", ""),
    "insufficient_funds_ask": ("Insufficient coin balance to sell", ""),
    "under_min_total_bid": ("Order amount is below the minimum order amount (5,000 KRW)", ""),
    "under_min_total_ask": ("Order amount is below the minimum order amount (5,000 KRW)", ""),
    "validation_error": ("Invalid request format", "Check settings, including the market name."),
    "invaild_parameter": ("Invalid request parameters", "Check configuration values."),
}


def _b64url(data: bytes) -> str:
    return base64.urlsafe_b64encode(data).rstrip(b"=").decode()


def _public_ip() -> str:
    try:
        return requests.get("https://api.ipify.org", timeout=3).text.strip()
    except requests.RequestException:
        return ""


class Upbit:
    def __init__(self, access_key: str = "", secret_key: str = "", timeout: float = 10.0):
        self.access_key = access_key
        self.secret_key = secret_key
        self.timeout = timeout
        self.server_time: datetime | None = None


    def _jwt(self, params: dict[str, Any] | None = None) -> str:
        payload: dict[str, Any] = {"access_key": self.access_key, "nonce": str(uuid.uuid4())}
        if params:
            qs = urlencode(params)          # Hash the exact query string sent to Upbit.
            payload["query_hash"] = hashlib.sha512(qs.encode()).hexdigest()
            payload["query_hash_alg"] = "SHA512"
        header = _b64url(json.dumps({"alg": "HS512", "typ": "JWT"}, separators=(",", ":")).encode())
        body = _b64url(json.dumps(payload, separators=(",", ":")).encode())
        sig = _b64url(hmac.new(self.secret_key.encode(), f"{header}.{body}".encode(),
                               hashlib.sha512).digest())
        return f"{header}.{body}.{sig}"

    def _raise(self, r: HttpResponse) -> None:
        name = ""
        with contextlib.suppress(ValueError):
            name = str(r.json().get("error", {}).get("name") or "")
        if name in _ERRORS:
            desc, hint = _ERRORS[name]
            if name == "no_authorization_ip":
                ip = _public_ip()
                hint += f" (Current public IP: {ip})" if ip else ""
            raise UpbitError(desc, hint, name, r.status_code)
        if r.status_code in (418, 429):
            raise UpbitError("Upbit temporarily rate-limited the request",
                             "Will retry automatically shortly.", name, r.status_code)
        if r.status_code >= 500:
            raise UpbitError("Upbit server issue (possibly maintenance)",
                             "Will retry automatically next cycle.", name, r.status_code)
        if r.status_code in (401, 403):
            raise UpbitError(f"API key authentication failed ({name or 'HTTP ' + str(r.status_code)})",
                             "Check UPBIT_ACCESS_KEY / UPBIT_SECRET_KEY in .env, plus the registered "
                             "IP address and API permissions (asset inquiry and orders).", name, r.status_code)
        raise UpbitError(f"Upbit rejected the request (HTTP {r.status_code}"
                         f"{', ' + name if name else ''})",
                         f"Response: {r.text[:200]}", name, r.status_code)

    def _request(self, method: str, path: str, params: dict[str, Any] | None = None,
                 private: bool = False, retries: int = 3) -> Any:
        url = BASE_URL + path
        last: Exception | None = None
        for attempt in range(retries):
            try:
                headers = {"Authorization": f"Bearer {self._jwt(params)}"} if private else {}
                if method == "GET":
                    r = requests.get(url, params=params, headers=headers, timeout=self.timeout)
                else:
                    headers["Content-Type"] = "application/json; charset=utf-8"
                    r = requests.post(url, json=params, headers=headers, timeout=self.timeout)
                if r.headers.get("Date"):
                    with contextlib.suppress(ValueError, TypeError):
                        self.server_time = pd.to_datetime(
                            r.headers["Date"], utc=True).to_pydatetime()
                if r.status_code >= 400:
                    self._raise(r)
                return r.json()
            except UpbitError as e:
                last = e
                if e.status in (418, 429) or e.status >= 500:
                    time.sleep(1.5 * (attempt + 1))
                    continue
                raise
            except requests.Timeout:
                last = UpbitError("Upbit request timed out", "Check your internet connection.")
                time.sleep(1.5 * (attempt + 1))
            except requests.RequestException:
                last = UpbitError("Unable to connect to Upbit",
                                  "Check your internet connection; processing resumes automatically when connected.")
                time.sleep(1.5 * (attempt + 1))
        raise last if last else UpbitError("Unknown communication error")

    def now_utc(self) -> datetime:
        t = self.server_time or datetime.now(UTC)
        return t if t.tzinfo else t.replace(tzinfo=UTC)


    def candles_1h(self, market: str, count: int) -> pd.DataFrame:
        """Completed hourly bars indexed by UTC opening timestamps."""
        frames: list[pd.DataFrame] = []
        to: str | None = None
        got = 0
        while got < count:
            params: dict[str, Any] = {"market": market, "count": min(CANDLE_PAGE, count - got)}
            if to:
                params["to"] = to
            rows = self._request("GET", "/v1/candles/minutes/60", params)
            if not rows:
                break
            df = self._rows_to_df(rows)
            frames.append(df)
            got += len(df)
            oldest = df.index.min()
            if len(rows) < params["count"]:
                break
            to = oldest.strftime("%Y-%m-%dT%H:%M:%SZ")     # Exclusive to boundary prevents duplicate or missing bars.
            time.sleep(PAGE_SLEEP)
        if not frames:
            raise UpbitError(f"{market} market data unavailable", "Check the market name.")
        out = pd.concat(frames)
        out = out[~out.index.duplicated(keep="last")].sort_index()
        now = pd.Timestamp(self.now_utc())
        return out[out.index + pd.Timedelta(hours=1) <= now]

    @staticmethod
    def _rows_to_df(rows: list[dict[str, Any]]) -> pd.DataFrame:
        df = pd.DataFrame(rows)
        df["ts"] = pd.to_datetime(df["candle_date_time_utc"], errors="coerce").dt.tz_localize(UTC)
        df = df.rename(columns={"opening_price": "open", "high_price": "high", "low_price": "low",
                                "trade_price": "close", "candle_acc_trade_volume": "volume"})
        df = df.dropna(subset=["ts"]).set_index("ts").sort_index()
        return df[["open", "high", "low", "close", "volume"]].astype(float)

    def last_price(self, market: str) -> float | None:
        try:
            j = self._request("GET", "/v1/ticker", {"markets": market})
            return float(j[0]["trade_price"])
        except (UpbitError, KeyError, IndexError, ValueError):
            return None


    def balances(self, market: str) -> tuple[float, float]:
        """Available KRW and total coin quantity, including locked coins."""
        coin = market.split("-")[1]
        rows: list[dict[str, Any]] = self._request("GET", "/v1/accounts", private=True)
        krw = qty = 0.0
        for r in rows:
            cur = r.get("currency")
            if cur == "KRW":
                krw = float(r.get("balance") or 0.0)
            elif cur == coin:
                qty = float(r.get("balance") or 0.0) + float(r.get("locked") or 0.0)
        return krw, qty


    def _order(self, params: dict[str, Any]) -> str:
        """Return the order ID or an empty string if submission is unresolved."""
        try:
            res = self._request("POST", "/v1/orders", params, private=True, retries=2)
        except UpbitError as e:
            if e.code == "duplicated_identifier":
                return ""
            raise
        return str(res.get("uuid") or "")

    def buy_market(self, market: str, krw: float, identifier: str) -> str:
        return self._order({"market": market, "side": "bid", "ord_type": "price",
                            "price": str(krw), "identifier": identifier})

    def sell_market(self, market: str, qty: float, identifier: str) -> str:
        return self._order({"market": market, "side": "ask", "ord_type": "market",
                            "volume": f"{qty:.8f}", "identifier": identifier})

    def order_state(self, identifier: str) -> dict[str, Any] | None:
        try:
            return self._request("GET", "/v1/order", {"identifier": identifier}, private=True,
                                 retries=2)
        except UpbitError:
            return None

    def wait_done(self, identifier: str, timeout_sec: float = 30.0) -> tuple[str, float, float]:
        """Return status, filled quantity and average price, including partial fills at timeout."""
        deadline = time.monotonic() + timeout_sec
        state, qty, avg = "unknown", 0.0, 0.0
        while time.monotonic() < deadline:
            j = self.order_state(identifier)
            if j:
                state = str(j.get("state") or "")
                qty = float(j.get("executed_volume") or 0.0)
                paid = float(j.get("price") or 0.0)
                trades = j.get("trades") or []
                if trades:
                    total = sum(float(t["price"]) * float(t["volume"]) for t in trades)
                    vol = sum(float(t["volume"]) for t in trades)
                    avg = total / vol if vol else 0.0
                elif qty > 0 and paid > 0:
                    avg = paid / qty
                if state in ("done", "cancel"):
                    return state, qty, avg
            time.sleep(1.5)
        return state, qty, avg
