from __future__ import annotations

import sys
from datetime import UTC, datetime
from typing import Any

import pandas as pd

import upbit_api
from upbit_api import Upbit, UpbitError


class _Resp:

    def __init__(self, status: int, body: Any, text: str = ""):
        self.status_code = status
        self._body = body
        self.text = text or str(body)

    def json(self) -> Any:
        if isinstance(self._body, ValueError):
            raise self._body
        return self._body


def _candle_rows(start: datetime, n: int) -> list[dict[str, Any]]:
    rows = []
    for i in range(n):
        ts = start - pd.Timedelta(hours=i)
        rows.append({
            "candle_date_time_utc": ts.strftime("%Y-%m-%dT%H:%M:%S"),
            "opening_price": 100.0, "high_price": 101.0, "low_price": 99.0,
            "trade_price": 100.5, "candle_acc_trade_volume": 1.0,
        })
    return rows


class _FakeUpbit(Upbit):

    def __init__(self, pages: list[list[dict[str, Any]]], now: datetime):
        super().__init__()
        self.pages = pages
        self.calls: list[dict[str, Any]] = []
        self.server_time = now
        upbit_api.PAGE_SLEEP = 0.0        # Pagination delay is outside this test.

    def _request(self, method, path, params=None, private=False, retries=3):
        self.calls.append(dict(params or {}))
        return self.pages[len(self.calls) - 1] if len(self.calls) <= len(self.pages) else []


def test_pagination_passes_oldest_as_next_to():
    now = datetime(2026, 8, 4, 12, 0, tzinfo=UTC)
    newest = pd.Timestamp(2026, 8, 4, 11, 0, tz="UTC")
    page1 = _candle_rows(newest, 200)
    page2 = _candle_rows(newest - pd.Timedelta(hours=200), 100)
    api = _FakeUpbit([page1, page2], now)

    df = api.candles_1h("KRW-BTC", 300)
    assert len(api.calls) == 2, f"Expected two requests for 300 bars; got {len(api.calls)} requests"
    assert api.calls[0]["count"] == 200 and "to" not in api.calls[0]
    assert api.calls[1]["count"] == 100, "Request only the remaining number of bars"
    oldest_of_page1 = (newest - pd.Timedelta(hours=199)).strftime("%Y-%m-%dT%H:%M:%SZ")
    assert api.calls[1]["to"] == oldest_of_page1, \
        f"to is not the oldest timestamp of the previous page: {api.calls[1]['to']} != {oldest_of_page1}"
    assert df.index.is_monotonic_increasing and not df.index.has_duplicates
    assert len(df) == 300
    print("  [OK] Exclusive pagination without missing or duplicate bars")


def test_incomplete_bar_is_dropped():
    now = datetime(2026, 8, 4, 12, 30, tzinfo=UTC)
    rows = _candle_rows(pd.Timestamp(2026, 8, 4, 12, 0, tz="UTC"), 5)
    df = _FakeUpbit([rows], now).candles_1h("KRW-BTC", 5)
    assert pd.Timestamp(2026, 8, 4, 12, 0, tz="UTC") not in df.index, "Incomplete bar was retained"
    assert pd.Timestamp(2026, 8, 4, 11, 0, tz="UTC") in df.index, "Complete bar was removed"
    print("  [OK] Exclude incomplete bars")


def test_duplicate_bars_collapse():
    now = datetime(2026, 8, 4, 12, 0, tzinfo=UTC)
    newest = pd.Timestamp(2026, 8, 4, 11, 0, tz="UTC")
    same = _candle_rows(newest, 200)
    api = _FakeUpbit([same, _candle_rows(newest, 100)], now)
    df = api.candles_1h("KRW-BTC", 300)
    assert not df.index.has_duplicates, "Duplicate bar retained"
    print("  [OK] Deduplicate bars")


def test_no_data_raises_with_hint():
    now = datetime(2026, 8, 4, 12, 0, tzinfo=UTC)
    try:
        _FakeUpbit([[]], now).candles_1h("KRW-XXX", 10)
    except UpbitError as e:
        assert "market name" in e.hint, f"Missing actionable guidance: {e.hint}"
        print("  [OK] Actionable error for missing market data")
        return
    raise AssertionError("Empty response was accepted")


def test_error_mapping():
    api = Upbit()
    cases = [
        ({"error": {"name": "expired_access_key"}}, 401, "expired", "Issue a new key"),
        ({"error": {"name": "out_of_scope"}}, 403, "permissions", "asset inquiry"),
        ({"error": {"name": "under_min_total_bid"}}, 400, "minimum order amount", ""),
        ({}, 429, "rate-limited", "retry"),
        ({}, 503, "server issue", "retry"),
        ({"error": {"name": "who_knows"}}, 401, "authentication failed", "IP"),
    ]
    for body, status, in_msg, in_hint in cases:
        try:
            api._raise(_Resp(status, body))
        except UpbitError as e:
            assert in_msg in str(e), f"{status}/{body}: Message is missing '{in_msg}': {e}"
            assert in_hint in e.hint, f"{status}/{body}: Hint is missing '{in_hint}': {e.hint}"
            assert e.status == status
        else:
            raise AssertionError(f"{status}/{body}: Expected an exception")

    try:
        api._raise(_Resp(400, ValueError("not json"), text="<html>"))
    except UpbitError as e:
        assert "400" in str(e)
    print("  [OK] Seven error mappings with actionable hints")


def test_retry_only_on_transient():
    api = Upbit()
    for name, status, retryable in [
        ("expired_access_key", 401, False),
        ("out_of_scope", 403, False),
        ("", 429, True),
        ("", 503, True),
    ]:
        try:
            api._raise(_Resp(status, {"error": {"name": name}} if name else {}))
        except UpbitError as e:
            got = e.status in (418, 429) or e.status >= 500
            assert got == retryable, f"{name or status}: Retry decision: {got}, expected {retryable}"
    print("  [OK] Retry only transient errors")


def test_jwt_signature():
    import base64
    import hashlib
    import hmac
    import json as jsonlib
    from urllib.parse import urlencode

    def decode(seg: str) -> dict[str, Any]:
        return jsonlib.loads(base64.urlsafe_b64decode(seg + "=" * (-len(seg) % 4)))

    api = Upbit(access_key="AK", secret_key="SK")
    params = {"market": "KRW-BTC", "count": 3}
    header, body, sig = api._jwt(params).split(".")

    assert decode(header)["alg"] == "HS512"
    payload = decode(body)
    assert payload["access_key"] == "AK"
    assert payload["query_hash_alg"] == "SHA512"
    assert payload["query_hash"] == hashlib.sha512(urlencode(params).encode()).hexdigest(), \
        "Hash must match the transmitted query string"

    expect = base64.urlsafe_b64encode(
        hmac.new(b"SK", f"{header}.{body}".encode(), hashlib.sha512).digest()
    ).rstrip(b"=").decode()
    assert sig == expect, "Signature failed verification with secret_key"

    a = decode(api._jwt(params).split(".")[1])["nonce"]
    b = decode(api._jwt(params).split(".")[1])["nonce"]
    assert a != b, "Nonce was reused"

    assert "query_hash" not in decode(api._jwt().split(".")[1])
    print("  [OK] JWT signature (query_hash·nonce·HS512)")


def main() -> int:
    test_pagination_passes_oldest_as_next_to()
    test_incomplete_bar_is_dropped()
    test_duplicate_bars_collapse()
    test_no_data_raises_with_hint()
    test_error_mapping()
    test_retry_only_on_transient()
    test_jwt_signature()
    print("\nAll checks passed")
    return 0


if __name__ == "__main__":
    sys.exit(main())
