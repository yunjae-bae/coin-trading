# -*- coding: utf-8 -*-
from __future__ import annotations

from typing import Optional

import requests

TIMEOUT = 10.0


def _send_telegram(token: str, chat_id: str, text: str) -> bool:
    try:
        r = requests.post(
            f"https://api.telegram.org/bot{token}/sendMessage",
            json={"chat_id": chat_id, "text": text, "disable_web_page_preview": True},
            timeout=TIMEOUT)
        return r.status_code == 200
    except requests.RequestException:
        return False


def send(cfg, text: str) -> bool:
    """Return False if unconfigured or unreachable; suppress transport failures."""
    if cfg.telegram_token and cfg.telegram_chat_id:
        return _send_telegram(cfg.telegram_token, cfg.telegram_chat_id, text)
    return False


def check(cfg) -> Optional[str]:
    """Return a configuration or delivery error, or None."""
    if not (cfg.telegram_token and cfg.telegram_chat_id):
        return "Notifications not configured"
    try:
        r = requests.get(f"https://api.telegram.org/bot{cfg.telegram_token}/getMe", timeout=TIMEOUT)
        if r.status_code != 200:
            return "Invalid Telegram bot token (TELEGRAM_TOKEN check)"
    except requests.RequestException:
        return "Unable to connect to Telegram (check internet connection)"
    if not send(cfg, "✅ Trading notifications connected."):
        return ("Invalid Telegram chat ID (TELEGRAM_CHAT_ID check).\n"
                "  Send a message to the bot first, then retry.")
    return None
