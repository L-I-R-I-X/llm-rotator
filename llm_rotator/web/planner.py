"""Планировщик хода: продолжить удалённый чат дельтой или начать новый.

Правило одно и строгое (оно же — исправление R2): дельта уходит в существующий
чат только если **доказано**, что чат уже содержит начало этого разговора, то
есть ключи отправленного — строгий префикс ключей запроса. Запрос, который
просто выглядит как «новое сообщение», новым разговором и считается: дописывать
его в последний чат нельзя, он отвечал бы про прошлый проект.
"""

from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
from dataclasses import dataclass

from ..registry import WebSettings
from .fingerprint import (
    Fingerprint,
    content_text,
    fingerprint,
    is_system,
    tool_calls_of,
    tool_names,
)
from .store import BUSY, ChatRecord, ConversationStore


@dataclass(frozen=True, slots=True)
class TurnPlan:
    kind: str
    """`delta` — писать в `chat`; `full` — завести новый чат."""
    prompt: str
    full_prompt: str
    """Полный промпт: понадобится, если чат вдруг пропал (`chat_gone`)."""
    fingerprint: Fingerprint
    chat: ChatRecord | None = None
    ephemeral: bool = False
    """Параллельный ход того же разговора: чат после ответа не оставляем."""


def render_system(messages: Sequence[Mapping]) -> str:
    parts = [
        content_text(m.get("content"))
        for m in messages
        if is_system(m) and m.get("content")
    ]
    return "\n".join(p for p in parts if p).strip()


def _tool_call_text(name: str, args: object) -> str:
    return json.dumps(
        {"tool_call": {"name": name, "arguments": args}}, ensure_ascii=False
    )


def render_conversation(
    messages: Sequence[Mapping], names: Mapping[str, str] | None = None
) -> str:
    """Реплики в виде транскрипта (порт `formatConversation`)."""
    names = names if names is not None else tool_names(messages)
    out: list[str] = []
    for message in messages:
        role = message.get("role")
        if is_system(message):
            continue
        text = content_text(message.get("content"))
        if role == "user":
            if text:
                out.append(f"User: {text}")
        elif role == "assistant":
            if text:
                out.append(f"Assistant: {text}")
            for name, args in tool_calls_of(message):
                out.append(f"Assistant: {_tool_call_text(name, args)}")
        elif role in ("tool", "function"):
            name = message.get("name") or names.get(
                str(message.get("tool_call_id")), ""
            )
            label = f": {name}" if name else ""
            out.append(f"[Tool Result{label}]\n{text}")
    return "\n\n".join(out).strip()


def join_prompt(system: str, conversation: str) -> str:
    return f"{system}\n\n{conversation}" if system else conversation


def plan_turn(
    store: ConversationStore,
    settings: WebSettings,
    *,
    provider: str,
    key_name: str,
    namespace: str,
    session: str,
    messages: Sequence[Mapping],
    tools: object = None,
) -> TurnPlan:
    fp = fingerprint(messages, tools)
    names = tool_names(messages)
    body = [m for m in messages if not is_system(m)]
    full = join_prompt(render_system(messages), render_conversation(body, names))

    def plan(kind: str, prompt: str = full, **extra) -> TurnPlan:
        return TurnPlan(kind, prompt, full, fp, **extra)

    best: ChatRecord | None = None
    request = fp.request_keys
    for record in store.candidates(provider, key_name, namespace, session):
        sent = record.sent_keys
        n = len(sent)
        if record.context_key != fp.context_key or not 0 < n < len(request):
            continue
        if request[:n] != sent:
            continue
        # В хвосте должно быть хоть что-то, кроме эха нашего же ответа.
        if all(role == "assistant" for role in fp.roles[n:]):
            continue
        if best is None or n > len(best.sent_keys):
            best = record
    if best is None:
        return plan("full")
    if best.state == BUSY:
        return plan("full", ephemeral=True)
    if (
        best.turns >= settings.chat_max_turns
        or best.remote_chars >= settings.chat_max_chars
    ):
        return plan("full")
    if not store.claim(provider, best.chat_id):
        return plan("full", ephemeral=True)
    tail = render_conversation(body[len(best.sent_keys) :], names)
    return plan("delta", tail, chat=best)
