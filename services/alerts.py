import os
import time
from typing import Callable, Dict

import requests


def send_telegram(text: str) -> None:
    """Send a Telegram message using bot credentials from environment.

    Environment variables:
        TELEGRAM_BOT_TOKEN: Bot token.
        TELEGRAM_CHAT_ID: Chat ID of the recipient.
    """
    token = os.getenv("TELEGRAM_BOT_TOKEN")
    chat_id = os.getenv("TELEGRAM_CHAT_ID")
    if not token or not chat_id:
        raise EnvironmentError("TELEGRAM_BOT_TOKEN and TELEGRAM_CHAT_ID must be set")

    url = f"https://api.telegram.org/bot{token}/sendMessage"
    payload = {"chat_id": chat_id, "text": text}
    requests.post(url, json=payload, timeout=10)


class AlertManager:
    """Manage alert notifications with cooldown control."""

    def __init__(self, channel: str, cooldown_sec: float) -> None:
        self.cooldown_sec = cooldown_sec
        self._last_sent: Dict[str, float] = {}
        self._channels: Dict[str, Callable[[str], None]] = {
            "telegram": send_telegram,
            "noop": lambda text: None,
        }
        if channel not in self._channels:
            raise ValueError(f"Unknown alert channel: {channel}")
        self._send = self._channels[channel]

    def notify(self, key: str, text: str) -> None:
        """Send `text` if `cooldown_sec` has passed for `key`."""
        now = time.time()
        last_time = self._last_sent.get(key)
        if last_time is not None and now - last_time < self.cooldown_sec:
            return
        self._last_sent[key] = now
        self._send(text)
