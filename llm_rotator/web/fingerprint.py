"""Отпечатки сообщений: как опознать продолжение разговора, а не угадать его.

Клиент каждый раз присылает весь разговор. Чтобы понять, что удалённый чат
уже содержит его начало, сравниваем не тексты, а короткие ключи нормализованных
сообщений. Нормализация убирает то, что клиенты переписывают при пересылке:
id вызовов инструментов, рассуждения, пробелы по краям.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping, Sequence
from dataclasses import dataclass

SYSTEM_ROLES = frozenset({"system", "developer"})
KEY_LENGTH = 16


def is_system(message: Mapping) -> bool:
    return message.get("role") in SYSTEM_ROLES


def _digest(value: object) -> str:
    raw = json.dumps(value, sort_keys=True, ensure_ascii=False, separators=(",", ":"))
    return hashlib.sha256(raw.encode()).hexdigest()


def has_images(message: Mapping) -> bool:
    content = message.get("content")
    return isinstance(content, list) and any(
        isinstance(part, Mapping) and part.get("type") in ("image_url", "image")
        for part in content
    )


def content_text(content: object) -> str:
    """Текст сообщения: части склеены, картинки заменены хешем."""
    if content is None:
        return ""
    if isinstance(content, str):
        return content.strip()
    if isinstance(content, Sequence):
        parts: list[str] = []
        for part in content:
            if isinstance(part, str):
                parts.append(part)
            elif isinstance(part, Mapping):
                if part.get("type") in ("image_url", "image"):
                    image = part.get("image_url")
                    url = image.get("url") if isinstance(image, Mapping) else image
                    parts.append(f"[image:{_digest(str(url))[:12]}]")
                elif isinstance(part.get("text"), str):
                    parts.append(part["text"])
        return "".join(parts).strip()
    return str(content).strip()


def canonical_json(value: object) -> str:
    return json.dumps(value, sort_keys=True, ensure_ascii=False, separators=(",", ":"))


def _arguments(raw: object) -> object:
    if isinstance(raw, str):
        try:
            return json.loads(raw) if raw.strip() else {}
        except ValueError:
            return raw
    return raw if raw is not None else {}


def tool_calls_of(message: Mapping) -> list[tuple[str, object]]:
    """Вызовы инструментов ассистента как (имя, аргументы), без id."""
    out: list[tuple[str, object]] = []
    for call in message.get("tool_calls") or ():
        if not isinstance(call, Mapping):
            continue
        function = call.get("function") or call
        out.append(
            (str(function.get("name") or ""), _arguments(function.get("arguments")))
        )
    return out


def tool_names(messages: Sequence[Mapping]) -> dict[str, str]:
    """id вызова → имя инструмента, чтобы узнавать результаты без id."""
    names: dict[str, str] = {}
    for message in messages:
        if message.get("role") != "assistant":
            continue
        for call in message.get("tool_calls") or ():
            if isinstance(call, Mapping) and call.get("id"):
                function = call.get("function") or call
                names[str(call["id"])] = str(function.get("name") or "")
    return names


def normalize(message: Mapping, names: Mapping[str, str]) -> dict:
    role = str(message.get("role") or "")
    if role == "function":
        role = "tool"
    normal: dict = {"role": role, "text": content_text(message.get("content"))}
    if role == "assistant":
        calls = tool_calls_of(message)
        if calls:
            normal["calls"] = [[name, canonical_json(args)] for name, args in calls]
    elif role == "tool":
        normal["tool"] = message.get("name") or names.get(
            str(message.get("tool_call_id") or ""), ""
        )
    return normal


def message_key(message: Mapping, names: Mapping[str, str] | None = None) -> str:
    return _digest(normalize(message, names or {}))[:KEY_LENGTH]


def reply_key(text: str, calls: Sequence[tuple[str, object]] = ()) -> str:
    """Ключ ответа ассистента, который мы вернули клиенту."""
    message: dict = {"role": "assistant", "content": text}
    if calls:
        message["tool_calls"] = [
            {"function": {"name": name, "arguments": args}} for name, args in calls
        ]
    return message_key(message)


def tools_description(tools: object) -> object:
    """Описание инструментов без шума: имя, описание, схема параметров."""
    if not tools:
        return []
    out = []
    for tool in tools:
        function = tool.get("function", tool) if isinstance(tool, Mapping) else {}
        out.append(
            {
                "name": function.get("name"),
                "description": function.get("description"),
                "parameters": function.get("parameters"),
            }
        )
    return out


@dataclass(frozen=True, slots=True)
class Fingerprint:
    """Что известно о запросе для сравнения с чатами."""

    request_keys: tuple[str, ...]
    """Ключи сообщений без системных, по порядку."""
    roles: tuple[str, ...]
    context_key: str
    """Системные сообщения и описание инструментов."""

    @property
    def first_key(self) -> str:
        return self.request_keys[0] if self.request_keys else ""


def fingerprint(messages: Sequence[Mapping], tools: object = None) -> Fingerprint:
    names = tool_names(messages)
    body = [m for m in messages if not is_system(m)]
    system = [content_text(m.get("content")) for m in messages if is_system(m)]
    return Fingerprint(
        request_keys=tuple(message_key(m, names) for m in body),
        roles=tuple(
            "tool" if m.get("role") == "function" else str(m.get("role")) for m in body
        ),
        context_key=_digest({"system": system, "tools": tools_description(tools)})[
            :KEY_LENGTH
        ],
    )
