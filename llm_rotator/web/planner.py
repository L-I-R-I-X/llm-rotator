"""Планировщик хода: продолжить удалённый чат дельтой или начать новый.

Правило одно и строгое (оно же — исправление R2): дельта уходит в существующий
чат только если **доказано**, что чат уже содержит начало этого разговора, то
есть ключи отправленного — строгий префикс ключей запроса. Запрос, который
просто выглядит как «новое сообщение», новым разговором и считается: дописывать
его в последний чат нельзя, он отвечал бы про прошлый проект.

Не влезающее в сообщение (R4) уходит файлом: история — в `context.md`, крупные
реплики последнего хода — отдельными файлами; в самом сообщении остаётся
инструкция, последний ход и пометки о приложенном.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping, Sequence
from dataclasses import dataclass

from ..registry import WebSettings
from .attachments import Attachment, collect_images, text_attachment
from .fingerprint import (
    Fingerprint,
    content_text,
    fingerprint,
    is_system,
    tool_calls_of,
    tool_names,
)
from .store import BUSY, ChatRecord, ConversationStore
from .tools import format_tool_definitions, format_tool_reminder

BIG_MESSAGE_CHARS = 20_000
"""Реплика последнего хода крупнее этого уходит отдельным файлом (если контекст
в сообщение не влезает)."""
CONTEXT_FILE = "context.md"
TRUNCATION_MARKER = (
    "\n\n[... middle of the text truncated to fit the file limit ...]\n\n"
)
CONTEXT_NOTE = (
    "[The earlier conversation is in the attached file context.md, in the same "
    "User:/Assistant: transcript format, oldest first. Read it before answering. "
    "The latest messages follow.]"
)


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
    attachments: tuple[Attachment, ...] = ()
    """Файлы и картинки этого хода (грузятся через связку)."""
    full_attachments: tuple[Attachment, ...] = ()
    too_large: str = ""
    """Причина отказа, если даже с файлами сообщение не влезает."""
    full_reason: str = ""
    """То же для полного промпта (запасной путь при `chat_gone`)."""

    def full_too_large(self) -> bool:
        return bool(self.full_reason)


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
    messages: Sequence[Mapping],
    names: Mapping[str, str] | None = None,
    hide_images: bool = False,
) -> str:
    """Реплики в виде транскрипта (порт `formatConversation`)."""
    names = names if names is not None else tool_names(messages)
    out: list[str] = []
    for message in messages:
        role = message.get("role")
        if is_system(message):
            continue
        text = content_text(message.get("content"), hide_images)
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


def fit(text: str, max_chars: int, head_ratio: float = 0.35) -> str:
    """Обрезать середину (порт `truncatePromptMiddle`), начало и конец сохранить."""
    if len(text) <= max_chars:
        return text
    payload = max_chars - len(TRUNCATION_MARKER)
    if payload <= 0:
        return text[len(text) - max_chars :]
    head = int(payload * head_ratio)
    return text[:head] + TRUNCATION_MARKER + text[len(text) - (payload - head) :]


def last_turn_start(body: Sequence[Mapping]) -> int:
    """Где начинается последний ход: после последней реплики ассистента.

    Реплика ассистента с вызовом инструмента входит в ход: без неё результат
    инструмента остался бы без вопроса.
    """
    for index in range(len(body) - 1, -1, -1):
        message = body[index]
        if message.get("role") == "assistant":
            return index if tool_calls_of(message) else index + 1
    return 0


def _externalize(
    messages: Sequence[Mapping],
    names: Mapping[str, str],
    hide_images: bool,
) -> tuple[str, list[Attachment]]:
    """Реплики последнего хода; крупные содержимым уходят в файлы-вложения."""
    files: list[Attachment] = []
    rendered: list[Mapping] = []
    for message in messages:
        text = content_text(message.get("content"), hide_images)
        if message.get("role") != "assistant" and len(text) > BIG_MESSAGE_CHARS:
            # Имя с хешем: в одном чате файлы разных ходов не должны совпасть.
            digest = hashlib.sha256(text.encode()).hexdigest()[:8]
            name = f"attachment_{digest}.txt"
            files.append(text_attachment(name, text))
            message = {**message, "content": f"[Content attached as file {name}]"}
        rendered.append(message)
    return render_conversation(rendered, names, hide_images), files


def build_prompts(
    system: str,
    body: Sequence[Mapping],
    names: Mapping[str, str],
    settings: WebSettings,
    *,
    with_history: bool,
) -> tuple[str, tuple[Attachment, ...], str]:
    """Промпт, вложения и причина отказа (пусто — всё в порядке).

    `with_history=False` — в чате уже есть история (дельта): файл с историей не
    нужен, только вынос крупных реплик, если сообщение не влезает.
    """
    hide = not settings.images
    inline = join_prompt(system, render_conversation(body, names, hide))
    if len(inline) <= settings.inline_max_chars:
        return inline, (), ""
    start = last_turn_start(body) if with_history else 0
    history, last = body[:start], body[start:]
    last_text, files = _externalize(last, names, hide)
    attachments: list[Attachment] = []
    note = ""
    history_text = render_conversation(history, names, hide)
    if history_text:
        attachments.append(
            text_attachment(CONTEXT_FILE, fit(history_text, settings.file_max_chars))
        )
        note = CONTEXT_NOTE + "\n\n"
    attachments += [
        text_attachment(f.name, fit(f.data.decode("utf-8"), settings.file_max_chars))
        for f in files
    ]
    prompt = join_prompt(system, note + last_text)
    reason = ""
    if len(prompt) > settings.inline_max_chars:
        reason = (
            f"даже с файлами сообщение ({len(prompt)} символов) больше "
            f"inline_max_chars ({settings.inline_max_chars})"
        )
    return prompt, tuple(attachments), reason


def find_continuation(
    store: ConversationStore,
    settings: WebSettings,
    *,
    provider: str,
    key_name: str,
    namespace: str,
    session: str,
    fp: Fingerprint,
) -> ChatRecord | None:
    """Чат этого аккаунта, который доказуемо содержит начало разговора.

    Ничего не занимает: годится и для выбора связки (affinity), и для плана.
    """
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
    return best


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
    system = (
        render_system(messages)
        + format_tool_definitions(tools, settings.inline_max_chars)
    ).strip()
    full, full_files, full_reason = build_prompts(
        system, body, names, settings, with_history=True
    )
    pictures = collect_images(body) if settings.images else ()
    full_files = (*full_files, *pictures)

    def plan(kind: str, prompt: str, files, reason: str, **extra) -> TurnPlan:
        return TurnPlan(
            kind,
            prompt,
            full,
            fp,
            attachments=tuple(files),
            full_attachments=full_files,
            too_large=reason,
            full_reason=full_reason,
            **extra,
        )

    def whole(**extra) -> TurnPlan:
        return plan("full", full, full_files, full_reason, **extra)

    best = find_continuation(
        store,
        settings,
        provider=provider,
        key_name=key_name,
        namespace=namespace,
        session=session,
        fp=fp,
    )
    if best is None:
        return whole()
    if best.state == BUSY:
        return whole(ephemeral=True)
    if (
        best.turns >= settings.chat_max_turns
        or best.remote_chars >= settings.chat_max_chars
    ):
        return whole()
    if not store.claim(provider, best.chat_id):
        return whole(ephemeral=True)
    tail_messages = body[len(best.sent_keys) :]
    tail, files, reason = build_prompts(
        "", tail_messages, names, settings, with_history=False
    )
    reminder = format_tool_reminder(tools)
    if reminder:
        # Полная инструкция лежит в начале чата; без напоминания модель
        # постепенно уходит от формата и отвечает блоками кода.
        tail = f"{tail}\n\n{reminder}"
    pictures = collect_images(tail_messages) if settings.images else ()
    return plan("delta", tail, (*files, *pictures), reason, chat=best)
