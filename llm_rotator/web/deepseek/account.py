"""Файл сессии аккаунта DeepSeek Web: `token, cookie, hif_dliq, hif_leim, wasmUrl, baseUrl`."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

from ..errors import WebError


@dataclass(frozen=True)
class Account:
    """Сессия аккаунта. Секреты скрыты из repr, чтобы не попасть в логи."""

    token: str = field(repr=False)
    cookie: str = field(default="", repr=False)
    hif_dliq: str = field(default="", repr=False)
    hif_leim: str = field(default="", repr=False)
    wasm_url: str = ""
    base_url: str = ""


def load_account(path: str | Path) -> Account:
    """Прочитать файл сессии. Нет файла или токена — `auth` (нужен перелогин)."""
    try:
        raw = json.loads(Path(path).read_text(encoding="utf-8"))
    except (OSError, ValueError):
        raise WebError("auth", "файл сессии не читается или не JSON") from None
    token = raw.get("token") if isinstance(raw, dict) else None
    if not isinstance(token, str) or not token.strip():
        raise WebError("auth", "в файле сессии нет token")

    def text(name: str) -> str:
        value = raw.get(name, "")
        return value if isinstance(value, str) else ""

    return Account(
        token=token.strip(),
        cookie=text("cookie"),
        hif_dliq=text("hif_dliq"),
        hif_leim=text("hif_leim"),
        wasm_url=text("wasmUrl"),
        base_url=text("baseUrl"),
    )
