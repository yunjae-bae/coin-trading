from __future__ import annotations

import os
from dataclasses import dataclass, field

BASE_DIR = os.path.abspath(os.path.dirname(__file__))
ENV_PATH = os.path.join(BASE_DIR, ".env")

TIMEFRAMES = ("1h", "2h", "3h", "4h", "6h", "8h", "12h", "1d")
STRATEGIES = ("qqe", "qqe_mod")

# QQE MOD uses different first-QQE defaults from standard QQE (Mihkel00).
MOD_RSI_PERIOD = 6
MOD_FACTOR = 3.0


class ConfigError(Exception):
    """Invalid runtime configuration."""


def tf_hours(timeframe: str) -> int:
    tf = timeframe.strip().lower()
    if tf.endswith("d"):
        return int(tf[:-1] or 1) * 24
    if tf.endswith("h"):
        return int(tf[:-1] or 1)
    raise ConfigError(f"[Configuration error] TIMEFRAME '{timeframe}' is unknown "
                      f"({' / '.join(TIMEFRAMES)}; choose one)")


def read_env_file(path: str = ENV_PATH) -> dict[str, tuple[str, int]]:
    """Map keys to (value, source line number)."""
    out: dict[str, tuple[str, int]] = {}
    if not os.path.exists(path):
        return out
    with open(path, encoding="utf-8-sig") as f:
        for lineno, raw in enumerate(f, start=1):
            line = raw.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            key, _, val = line.partition("=")
            val = val.split("#", 1)[0].strip().strip('"').strip("'")
            out[key.strip().upper()] = (val, lineno)
    return out


@dataclass
class Config:
    market: str = "KRW-BTC"
    timeframe: str = "6h"
    strategy: str = "qqe_mod"
    position_pct: float = 50.0
    max_order_krw: float = 0.0            # Maximum KRW per buy; zero means unlimited.
    stop_loss_pct: float = 5.0
    daily_loss_limit_pct: float = 10.0
    cooldown_bars: int = 0
    price_guard_pct: float = 5.0          # Defer buys this far from the signal close; zero disables the guard.
    signal_max_age_min: float = 60.0
    # Zero disables the entry-age check.
    dry_run: bool = True
    telegram_token: str = ""
    telegram_chat_id: str = ""
    heartbeat_hour: int = 9        # KST hour for the daily heartbeat.
    access_key: str = ""
    secret_key: str = ""
    fee_rate: float = 0.0005
    min_order_krw: float = 5000.0
    slippage_pct: float = 0.05    # Backtest-only adverse execution slippage.
    rsi_period: int = 14
    sf: int = 5
    factor: float = 4.238
    rsi_period2: int = 6
    sf2: int = 5
    factor2: float = 1.61
    bb_length: int = 50
    bb_mult: float = 0.35
    threshold: float = 3.0
    source: str = field(default="", repr=False)

    @property
    def bar_hours(self) -> int:
        return tf_hours(self.timeframe)


def _get(env: dict[str, tuple[str, int]], key: str) -> tuple[str | None, int]:
    v = env.get(key)
    return (None, 0) if v is None else v


def _read_num(env: dict[str, tuple[str, int]], key: str, lo: float,
              hi: float) -> tuple[float, str, int] | None:
    """Return (value, source text, line number), or None if absent."""
    raw, lineno = _get(env, key)
    if raw is None or raw == "":
        return None
    try:
        val = float(raw)
    except ValueError:
        raise ConfigError(f"[Configuration error] {key} value '{raw}' is invalid; enter a number. "
                          f"(.env line {lineno})") from None
    if not (lo <= val <= hi):
        raise ConfigError(f"[Configuration error] {key} value {raw} is outside the allowed range({lo}~{hi}). "
                          f"(.env line {lineno})")
    return val, raw, lineno


def _float(env: dict[str, tuple[str, int]], key: str, default: float, lo: float,
           hi: float) -> float:
    got = _read_num(env, key, lo, hi)
    return default if got is None else got[0]


def _int(env: dict[str, tuple[str, int]], key: str, default: int, lo: float, hi: float) -> int:
    got = _read_num(env, key, lo, hi)
    if got is None:
        return default
    val, raw, lineno = got
    if val != int(val):
        raise ConfigError(f"[Configuration error] {key} must be an integer (received: {raw}). (.env line {lineno})")
    return int(val)


def _choice(env: dict[str, tuple[str, int]], key: str, default: str, allowed: tuple[str, ...]) -> str:
    raw, lineno = _get(env, key)
    if raw is None or raw == "":
        return default
    val = raw.strip().lower()
    if val not in allowed:
        raise ConfigError(f"[Configuration error] {key} value '{raw}' is not supported. "
                          f"{' or '.join(allowed)} Choose one of the listed values. (.env line {lineno})")
    return val


def load_config(env_path: str = ENV_PATH) -> Config:
    env = read_env_file(env_path)
    cfg = Config(
        market=(_get(env, "MARKET")[0] or "KRW-BTC").upper(),
        timeframe=_choice(env, "TIMEFRAME", "6h", TIMEFRAMES),
        strategy=_choice(env, "STRATEGY", "qqe", STRATEGIES),
        position_pct=_float(env, "POSITION_PCT", 50.0, 1.0, 100.0),
        max_order_krw=_float(env, "MAX_ORDER_KRW", 0.0, 0.0, 10_000_000_000.0),
        stop_loss_pct=_float(env, "STOP_LOSS_PCT", 5.0, 0.0, 90.0),
        daily_loss_limit_pct=_float(env, "DAILY_LOSS_LIMIT_PCT", 10.0, 0.0, 90.0),
        cooldown_bars=_int(env, "COOLDOWN_BARS", 0, 0, 100),
        price_guard_pct=_float(env, "PRICE_GUARD_PCT", 5.0, 0.0, 50.0),
        signal_max_age_min=_float(env, "SIGNAL_MAX_AGE_MIN", 60.0, 0.0, 10_080.0),
        dry_run=(_get(env, "DRY_RUN")[0] or "1").strip() != "0",
        telegram_token=(_get(env, "TELEGRAM_TOKEN")[0] or "").strip(),
        telegram_chat_id=(_get(env, "TELEGRAM_CHAT_ID")[0] or "").strip(),
        heartbeat_hour=_int(env, "HEARTBEAT_HOUR", 9, 0, 23),
        access_key=(_get(env, "UPBIT_ACCESS_KEY")[0] or "").strip(),
        secret_key=(_get(env, "UPBIT_SECRET_KEY")[0] or "").strip(),
        fee_rate=_float(env, "FEE_RATE", 0.0005, 0.0, 0.01),
        min_order_krw=_float(env, "MIN_ORDER_KRW", 5000.0, 1000.0, 1_000_000.0),
        slippage_pct=_float(env, "SLIPPAGE_PCT", 0.05, 0.0, 5.0),
        rsi_period=_int(env, "QQE_RSI_PERIOD", 14, 2, 200),
        sf=_int(env, "QQE_SF", 5, 1, 200),
        factor=_float(env, "QQE_FACTOR", 4.238, 0.1, 20.0),
        rsi_period2=_int(env, "QQE_RSI_PERIOD2", 6, 2, 200),
        sf2=_int(env, "QQE_SF2", 5, 1, 200),
        factor2=_float(env, "QQE_FACTOR2", 1.61, 0.1, 20.0),
        bb_length=_int(env, "QQE_BB_LENGTH", 50, 2, 500),
        bb_mult=_float(env, "QQE_BB_MULT", 0.35, 0.0, 10.0),
        threshold=_float(env, "QQE_THRESHOLD", 3.0, 0.0, 50.0),
        source=env_path if os.path.exists(env_path) else "(no .env; using defaults)",
    )
    if cfg.strategy == "qqe_mod":
        if "QQE_RSI_PERIOD" not in env:
            cfg.rsi_period = MOD_RSI_PERIOD
        if "QQE_FACTOR" not in env:
            cfg.factor = MOD_FACTOR
    if not cfg.dry_run and not (cfg.access_key and cfg.secret_key):
        raise ConfigError("[Configuration error] Live mode (DRY_RUN=0) requires API keys.\n"
                          "  Set UPBIT_ACCESS_KEY and UPBIT_SECRET_KEY in .env.")
    return cfg


def describe(cfg: Config) -> str:
    sl = f"{cfg.stop_loss_pct:g}%" if cfg.stop_loss_pct > 0 else "Disabled"
    dl = f"{cfg.daily_loss_limit_pct:g}%" if cfg.daily_loss_limit_pct > 0 else "Disabled"
    cap = f"{cfg.max_order_krw:,.0f} KRW" if cfg.max_order_krw > 0 else "Unlimited"
    mode = "Dry run (no orders)" if cfg.dry_run else "★LIVE ORDERS★"
    noti = "Telegram enabled" if (cfg.telegram_token and cfg.telegram_chat_id) else "★Disabled★"
    return (f"  Notifications  : {noti}\n"
            f"  Market         : {cfg.market}\n"
            f"  Timeframe      : {cfg.timeframe}\n"
            f"  Strategy       : {cfg.strategy}\n"
            f"  Allocation     : {cfg.position_pct:g}%  (Maximum buy amount {cap})\n"
            f"  Stop loss      : {sl}\n"
            f"  Daily loss limit: {dl}\n"
            f"  Mode           : {mode}\n"
            f"  Configuration file   : {cfg.source}")
