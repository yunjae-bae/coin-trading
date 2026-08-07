from __future__ import annotations

from typing import Protocol

import requests

from coin_config import Config

TIMEOUT = 10.0


class Unreachable(Exception):
    """Telegram transport failure."""


class Telegram(Protocol):

    def send_message(self, token: str, chat_id: str, text: str) -> bool:
        """Return delivery success; raise on invalid token or chat ID."""
        ...

    def token_valid(self, token: str) -> bool:
        ...


class HttpTelegram:
    def send_message(self, token: str, chat_id: str, text: str) -> bool:
        try:
            r = requests.post(
                f"https://api.telegram.org/bot{token}/sendMessage",
                json={"chat_id": chat_id, "text": text, "disable_web_page_preview": True},
                timeout=TIMEOUT)
        except requests.RequestException as e:
            raise Unreachable from e
        return r.status_code == 200

    def token_valid(self, token: str) -> bool:
        try:
            r = requests.get(f"https://api.telegram.org/bot{token}/getMe", timeout=TIMEOUT)
        except requests.RequestException as e:
            raise Unreachable from e
        return r.status_code == 200


DEFAULT: Telegram = HttpTelegram()


def send(cfg: Config, text: str, api: Telegram = DEFAULT) -> bool:
    """Return False if unconfigured or unreachable; suppress transport failures."""
    if not (cfg.telegram_token and cfg.telegram_chat_id):
        return False
    try:
        return api.send_message(cfg.telegram_token, cfg.telegram_chat_id, text)
    except Unreachable:
        return False


def check(cfg: Config, api: Telegram = DEFAULT) -> str | None:
    """Return a configuration or delivery error, or None."""
    if not (cfg.telegram_token and cfg.telegram_chat_id):
        return "Notifications not configured"
    try:
        if not api.token_valid(cfg.telegram_token):
            return "Invalid Telegram bot token (TELEGRAM_TOKEN check)"
        if not api.send_message(cfg.telegram_token, cfg.telegram_chat_id,
                                "✅ Trading notifications connected."):
            return ("Invalid Telegram chat ID (TELEGRAM_CHAT_ID check).\n"
                    "  Send a message to the bot first, then retry.")
    except Unreachable:
        return "Unable to connect to Telegram (check internet connection)"
    return None
