"""Send formatted trade signals to Telegram.

Used by the (future) live runner: when the strategy emits a :class:`Signal`,
format it and POST to the Telegram Bot API. Safe to import without ``requests``
installed -- sending simply raises a clear error in that case.
"""

from __future__ import annotations

from ..core.models import Signal


def format_signal(signal: Signal) -> str:
    s = signal
    arrow = "🟢 LONG" if s.direction.value == "long" else "🔴 SHORT"
    lines = [
        f"<b>{arrow}  {s.symbol}</b>",
        f"TF set: {s.tf_set}",
        f"Time: {s.time:%Y-%m-%d %H:%M %Z}",
        "",
        f"Entry : <code>{s.entry:.5f}</code>",
        f"SL    : <code>{s.stop_loss:.5f}</code>",
        f"TP    : <code>{s.take_profit:.5f}</code>",
        f"R:R   : <b>{s.rr:.2f}</b>",
    ]
    if s.crt is not None:
        lines.append(f"CRT   : {s.crt.low:.5f} – {s.crt.high:.5f}")
    if s.reason:
        lines.append("")
        lines.append(f"<i>{s.reason}</i>")
    return "\n".join(lines)


class TelegramNotifier:
    API = "https://api.telegram.org/bot{token}/sendMessage"

    def __init__(self, bot_token: str, chat_id: str, enabled: bool = True):
        self.bot_token = bot_token
        self.chat_id = chat_id
        self.enabled = enabled and bool(bot_token) and bool(chat_id)

    @classmethod
    def from_config(cls, cfg: dict) -> "TelegramNotifier":
        t = cfg.get("notify", {}).get("telegram", {})
        return cls(
            bot_token=t.get("bot_token", ""),
            chat_id=t.get("chat_id", ""),
            enabled=t.get("enabled", False),
        )

    def send(self, text: str) -> bool:
        if not self.enabled:
            return False
        try:
            import requests  # imported lazily so backtests don't need it
        except ImportError as exc:  # pragma: no cover
            raise RuntimeError("requests is required to send Telegram messages") from exc

        resp = requests.post(
            self.API.format(token=self.bot_token),
            json={
                "chat_id": self.chat_id,
                "text": text,
                "parse_mode": "HTML",
                "disable_web_page_preview": True,
            },
            timeout=15,
        )
        return resp.ok

    def send_signal(self, signal: Signal) -> bool:
        return self.send(format_signal(signal))
