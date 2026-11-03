"""Разбор SSE-ответа `/api/v0/chat/completion` в события.

Порт логики `server.js:2929-2990` (FreeDeepseekAPI). Ответ приходит строками
`data: {...}` с тремя видами полезной нагрузки:

- полный снимок `{"v": {"response": {"fragments": [...]}}}`;
- патч с путём `{"p": "response/fragments", "o": "APPEND", "v": [...]}`;
- короткая дозапись `{"v": "текст"}`, путь `p` у которой наследуется от
  предыдущей строки.

Тексту соответствуют фрагменты RESPONSE/SEARCH, рассуждению — THINK/REASONING.
"""

from __future__ import annotations

import codecs
import json
from collections.abc import Callable, Iterable, Iterator
from dataclasses import dataclass

from .. import errors
from ..errors import WebError

ANSWER_TYPES = frozenset({"RESPONSE", "SEARCH"})
REASONING_TYPES = frozenset({"THINK", "REASONING"})


@dataclass(frozen=True, slots=True)
class Ready:
    """Сервер принял запрос. `response_message_id` — parent для следующего хода."""

    response_message_id: int | str | None
    request_message_id: int | str | None = None


@dataclass(frozen=True, slots=True)
class Delta:
    text: str
    reasoning: bool = False


@dataclass(frozen=True, slots=True)
class Finish:
    """Конец ответа. `length` — обрезан по длине (INCOMPLETE)."""

    reason: str


Event = Ready | Delta | Finish


def split_sse(chunks: Iterable[bytes]) -> Iterator[tuple[str, str]]:
    """Байты → пары (имя события, data). UTF-8 собирается инкрементально."""
    decoder = codecs.getincrementaldecoder("utf-8")(errors="replace")
    buffer = ""
    name = ""
    data: list[str] = []

    def flush() -> tuple[str, str] | None:
        nonlocal name, data
        pair = (name or "message", "\n".join(data)) if data else None
        name, data = "", []
        return pair

    def lines(text: str) -> Iterator[str]:
        nonlocal buffer
        buffer += text
        *done, buffer = buffer.replace("\r\n", "\n").split("\n")
        yield from done

    def handle(line: str) -> tuple[str, str] | None:
        nonlocal name
        if not line:
            return flush()
        if line.startswith(":"):
            return None
        field, _, value = line.partition(":")
        value = value[1:] if value.startswith(" ") else value
        if field == "event":
            name = value
        elif field == "data":
            data.append(value)
        return None

    for chunk in chunks:
        for line in lines(decoder.decode(chunk)):
            pair = handle(line)
            if pair:
                yield pair
    for line in lines(decoder.decode(b"", final=True) + "\n"):
        pair = handle(line)
        if pair:
            yield pair
    pair = flush()
    if pair:
        yield pair


def error_from_hint(payload: dict) -> WebError:
    """Ошибка модели внутри HTTP 200: `{"type":"error","finish_reason":...}`."""
    reason = str(payload.get("finish_reason") or "")
    content = str(payload.get("content") or "")
    if reason == "backend_busy":
        return WebError("busy", content or reason)
    kind = errors.classify_text(f"{content} {reason}")
    if kind:
        return WebError(kind, content or reason)
    return WebError("upstream", content or reason or "ошибка модели")


class CompletionParser:
    """Конечный автомат разбора. `feed` принимает JSON-строки из `data:`."""

    def __init__(self) -> None:
        self._path: str | None = None
        self._fragments: list[dict] = []
        self._emitted: list[int] = []
        self._legacy_text = ""
        self._legacy_emitted = 0
        self._ready = False
        self.finish_reason = ""
        self.status = ""
        self.message_id: int | str | None = None
        self.text_chars = 0

    def feed(self, payload: str) -> list[Event]:
        payload = payload.strip()
        if not payload or payload == "[DONE]":
            return []
        try:
            data = json.loads(payload)
        except ValueError:
            return []
        if not isinstance(data, dict):
            return []

        events: list[Event] = []
        if data.get("type") == "error":
            raise error_from_hint(data)
        if "response_message_id" in data and not self._ready:
            self._ready = True
            self.message_id = data["response_message_id"]
            events.append(
                Ready(data["response_message_id"], data.get("request_message_id"))
            )
        if data.get("finish_reason"):
            self.finish_reason = str(data["finish_reason"])
        if "p" in data:
            self._path = data["p"]

        value = data.get("v")
        if isinstance(value, dict) and isinstance(value.get("response"), dict):
            self._apply_snapshot(value["response"])
        path = self._path
        if path == "response/fragments" and value is not None:
            self._append_fragments(value)
        elif path == "response" and isinstance(value, list):
            for op in value:
                if (
                    isinstance(op, dict)
                    and op.get("p") == "fragments"
                    and op.get("o") == "APPEND"
                    and op.get("v") is not None
                ):
                    self._append_fragments(op["v"])
        elif path == "response/fragments/-1/content" and isinstance(value, str):
            if self._fragments:
                last = self._fragments[-1]
                last["content"] = f"{last.get('content') or ''}{value}"
        elif path == "response/content" and isinstance(value, str):
            self._legacy_text += value
        elif path == "response/finish_reason" and value is not None:
            self.finish_reason = str(value)
        elif path == "response/status" and isinstance(value, str):
            self.status = value
        events.extend(self._new_text())
        return events

    def finish(self) -> Finish:
        reason = self.finish_reason
        if not reason and self.status.upper() == "INCOMPLETE":
            reason = "length"
        return Finish(reason or "stop")

    # --- внутреннее ---------------------------------------------------------

    def _apply_snapshot(self, response: dict) -> None:
        if response.get("message_id") is not None:
            self.message_id = response["message_id"]
        if isinstance(response.get("content"), str):
            self._legacy_text = response["content"]
        if isinstance(response.get("fragments"), list):
            self._fragments = []
            self._append_fragments(response["fragments"])
        if response.get("finish_reason") is not None:
            self.finish_reason = str(response["finish_reason"])
        if isinstance(response.get("status"), str):
            self.status = response["status"]

    def _append_fragments(self, value: object) -> None:
        for fragment in value if isinstance(value, list) else [value]:
            if isinstance(fragment, dict):
                self._fragments.append(dict(fragment))

    def _new_text(self) -> Iterator[Delta]:
        """Выдать только ещё не показанный хвост каждого фрагмента."""
        del self._emitted[len(self._fragments) :]
        self._emitted.extend([0] * (len(self._fragments) - len(self._emitted)))
        for index, fragment in enumerate(self._fragments):
            content = fragment.get("content")
            if not isinstance(content, str):
                continue
            kind = fragment.get("type")
            if kind not in ANSWER_TYPES and kind not in REASONING_TYPES:
                continue
            start = min(self._emitted[index], len(content))
            if len(content) > start:
                self._emitted[index] = len(content)
                if kind in ANSWER_TYPES:
                    self.text_chars += len(content) - start
                yield Delta(content[start:], reasoning=kind in REASONING_TYPES)
        if len(self._legacy_text) > self._legacy_emitted:
            text = self._legacy_text[self._legacy_emitted :]
            self._legacy_emitted = len(self._legacy_text)
            self.text_chars += len(text)
            yield Delta(text)


def iter_events(
    chunks: Iterable[bytes], on_event: Callable[[], None] | None = None
) -> Iterator[Event]:
    """Байты ответа → события. Ошибка в потоке — `WebError`.

    `on_event` зовётся на каждое SSE-событие (для таймера простоя).
    Пустой ответ без ошибки — `WebError("empty")`.
    """
    parser = CompletionParser()
    for name, data in split_sse(chunks):
        if on_event:
            on_event()
        if name == "close":
            break
        yield from parser.feed(data)
    if not parser.text_chars:
        raise WebError("empty", "ответ без текста")
    yield parser.finish()
