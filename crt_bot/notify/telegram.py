"""Send formatted trade signals to Telegram.

Used by the (future) live runner: when the strategy emits a :class:`Signal`,
format it and POST to the Telegram Bot API. Safe to import without ``requests``
installed -- sending simply raises a clear error in that case.
"""

from __future__ import annotations

from ..core.models import Signal

_TP_LABEL = {
    "crt_high": "CRT High (trend-aligned)",
    "crt_low": "CRT Low (trend-aligned)",
    "equilibrium": "50% equilibrium (counter-trend)",
}


def _fmt(x: float) -> str:
    """Price formatter that adapts decimals to the instrument's magnitude."""
    ax = abs(x)
    if ax >= 1000:
        dec = 2
    elif ax >= 1:
        dec = 4
    else:
        dec = 5
    return f"{x:,.{dec}f}"


def format_signal(
    signal: Signal,
    account_balance: float | None = None,
    risk_pct: float | None = None,
    session_name: str | None = None,
) -> str:
    """Full, self-contained signal card: entry, SL, TP, R:R, CRT context,
    market bias, TP rule, trigger and (optionally) a suggested position size."""
    s = signal
    head = "🟢 LONG" if s.direction.value == "long" else "🔴 SHORT"
    tf = s.tf_set.replace("-", " ▸ ")
    sess = f"  ·  {session_name} session" if session_name else ""

    lines = [
        f"<b>{head}  {s.symbol}</b>",
        f"⏱ {tf}{sess}",
        f"🕐 {s.time:%Y-%m-%d %H:%M} UTC",
        "",
        f"🎯 <b>Entry</b> : <code>{_fmt(s.entry)}</code>",
        f"🛑 <b>SL</b>    : <code>{_fmt(s.stop_loss)}</code>",
        f"✅ <b>TP</b>    : <code>{_fmt(s.take_profit)}</code>",
        f"📊 <b>R:R</b>   : <b>{s.rr:.2f}</b>",
    ]

    # suggested size from the risk model (size * SL-distance = risked amount)
    if account_balance and risk_pct and s.risk > 0:
        risk_amount = account_balance * risk_pct / 100.0
        size = risk_amount / s.risk
        lines.append(
            f"💰 Risk {risk_pct:g}% (≈ {risk_amount:,.2f})  →  size ≈ <b>{size:.4f}</b>"
        )

    lines.append("")
    if s.crt is not None:
        lines.append(f"📦 CRT range : {_fmt(s.crt.low)} – {_fmt(s.crt.high)}")
        lines.append(f"⚖️ 50% (eq)  : {_fmt(s.crt.equilibrium)}")
    if s.market_bias:
        basis = f" ({s.bias_basis})" if s.bias_basis else ""
        lines.append(f"📈 Market bias: <b>{s.market_bias.upper()}</b>{basis}")
    if s.tp_mode:
        lines.append(f"🎯 TP rule   : {_TP_LABEL.get(s.tp_mode, s.tp_mode)}")
    if s.entry_trigger:
        lines.append(f"🔑 Trigger   : {s.entry_trigger.upper()} after LTF CHoCH")

    lines.append("")
    lines.append("<i>✓ POI + CRT + MTF CHoCH + 0.618–0.786 pullback + LTF CHoCH + entry</i>")
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

    def send_signal(
        self,
        signal: Signal,
        account_balance: float | None = None,
        risk_pct: float | None = None,
        session_name: str | None = None,
    ) -> bool:
        return self.send(
            format_signal(signal, account_balance, risk_pct, session_name)
        )
