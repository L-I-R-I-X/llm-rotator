"""Ошибки веб-чатов и их перевод в `Failure` ротатора."""

from __future__ import annotations

import re
from dataclasses import dataclass

from ..gateway import Failure

KINDS: tuple[str, ...] = (
    "auth",
    "busy",
    "rate_limit",
    "chat_gone",
    "too_large",
    "bad_request",
    "upstream",
    # Сетевые, не из тела ответа:
    "timeout",
    "network",
    "blocked",
    "empty",
    # Не отказ аккаунта: слотов связки нет (`max_concurrent`), идём дальше.
    "busy_slot",
)
"""Виды ошибок. Первые семь — из плана, остальные — сеть, Cloudflare и пустой ответ."""

DETAIL_LIMIT = 300

RATE_LIMIT_RE = re.compile(
    r"too\s+many\s+(?:messages|requests)|rate[\s_-]?limit|过于频繁"
    r"|слишком\s+(?:много\s+(?:сообщений|запросов)|частые\s+сообщения)",
    re.IGNORECASE,
)
TOO_LARGE_RE = re.compile(
    r"(?:content|prompt|context).{0,40}(?:too\s+long|too\s+large|length|limit|maximum)"
    r"|maximum.{0,30}(?:context|token)|too\s+many\s+tokens"
    r"|содержани[ея]\s+слишком\s+длин|контекст.{0,30}(?:длин|лимит)",
    re.IGNORECASE,
)

RETRY_AFTER = {"rate_limit": 60.0, "busy": 30.0, "empty": 10.0, "upstream": 10.0}
"""Пауза связки по умолчанию, секунд (план, раздел 2.4)."""


@dataclass
class WebError(Exception):
    """Отказ веб-чата. `detail` — короткий текст без токенов и cookie."""

    kind: str
    detail: str = ""
    retry_after: float | None = None
    status: int | None = None

    def __post_init__(self) -> None:
        assert self.kind in KINDS, self.kind
        self.detail = " ".join(self.detail.split())[:DETAIL_LIMIT]
        Exception.__init__(self, f"{self.kind}: {self.detail}")

    def __str__(self) -> str:
        return f"{self.kind}: {self.detail}" if self.detail else self.kind


def classify_text(text: str) -> str | None:
    """Вид ошибки по тексту сообщения сервера, если он узнаваем."""
    if RATE_LIMIT_RE.search(text):
        return "rate_limit"
    if TOO_LARGE_RE.search(text):
        return "too_large"
    return None


def to_failure(error: WebError) -> tuple[Failure, float | None]:
    """Что считать сломанным и на сколько (Failure, retry_after), раздел 2.4.

    `chat_gone` отказом не является: транспорт сам заводит новый чат и
    повторяет в той же попытке, сюда он попадает только если это не помогло.
    """
    after = error.retry_after
    match error.kind:
        case "auth":
            return Failure.KEY, None
        case "rate_limit" | "busy" | "empty" | "upstream":
            default = RETRY_AFTER[error.kind]
            return Failure.QUOTA, after if after is not None else default
        case "timeout" | "network":
            return Failure.PROXY, None
        case "blocked":
            return Failure.BLOCKED, None
        case "too_large":
            return Failure.TOO_LARGE, None
        case "chat_gone":
            return Failure.QUOTA, after if after is not None else 0.0
        case _:  # bad_request
            return Failure.FATAL, None
