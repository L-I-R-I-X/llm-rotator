"""Формат OpenAI: разбор запроса и сборка ответов.

Всё, что знает про вид JSON наружу, лежит здесь: когда понадобится Anthropic
`/v1/messages`, рядом ляжет `anthropic_api.py`, а ядро и приложение не изменятся.
"""

from __future__ import annotations

import json
import time
import uuid
from collections.abc import Mapping
from dataclasses import dataclass, field

from ..transport import ChatResponse, ToolCall, Usage

PASSTHROUGH = (
    "temperature",
    "top_p",
    "stop",
    "seed",
    "presence_penalty",
    "frequency_penalty",
    "response_format",
    "tools",
    "tool_choice",
    "parallel_tool_calls",
)
"""Параметры, которые едут к провайдеру как есть. Остальное отбрасывается: чужое
поле провайдер встретит как 400, и обход его не вылечит."""


class RequestError(ValueError):
    """Неверный запрос клиента (400)."""


@dataclass(slots=True)
class ChatInput:
    messages: list[dict]
    model: str = ""
    stream: bool = False
    include_usage: bool = False
    session: str = ""
    params: dict[str, object] = field(default_factory=dict)


def parse_chat_request(body: object, header_session: str = "") -> ChatInput:
    if not isinstance(body, dict):
        raise RequestError("тело запроса должно быть JSON-объектом")
    messages = body.get("messages")
    if not isinstance(messages, list) or not messages:
        raise RequestError("поле 'messages' должно быть непустым списком")
    for index, message in enumerate(messages):
        if not isinstance(message, dict) or not isinstance(message.get("role"), str):
            raise RequestError(f"messages[{index}]: нужен объект с полем 'role'")
    params: dict[str, object] = {
        key: body[key] for key in PASSTHROUGH if body.get(key) is not None
    }
    limit = body.get("max_completion_tokens", body.get("max_tokens"))
    if limit is not None:
        if isinstance(limit, bool) or not isinstance(limit, int) or limit <= 0:
            raise RequestError("max_tokens должно быть положительным целым")
        params["max_tokens"] = limit
    if not params.get("tools"):
        params.pop("tools", None)
        params.pop("tool_choice", None)
        params.pop("parallel_tool_calls", None)
    options = body.get("stream_options")
    user = body.get("user")
    return ChatInput(
        messages=messages,
        model=str(body.get("model") or ""),
        stream=bool(body.get("stream")),
        include_usage=bool(isinstance(options, dict) and options.get("include_usage")),
        session=header_session or (user if isinstance(user, str) else ""),
        params=params,
    )


def new_id() -> str:
    return f"chatcmpl-{uuid.uuid4().hex[:24]}"


def estimate_usage(messages: list[dict], response: ChatResponse) -> Usage:
    """Веб-чаты токенов не считают: грубая оценка (4 символа на токен)."""
    if response.usage.total_tokens:
        return response.usage
    prompt = len(json.dumps(messages, ensure_ascii=False)) // 4
    answer = (len(response.text) + len(response.reasoning)) // 4
    answer += sum(len(c.raw_arguments) for c in response.tool_calls) // 4
    return Usage(prompt, answer, prompt + answer)


def usage_json(usage: Usage) -> dict:
    return {
        "prompt_tokens": usage.prompt_tokens,
        "completion_tokens": usage.completion_tokens,
        "total_tokens": usage.total_tokens,
    }


def tool_calls_json(calls: tuple[ToolCall, ...], indexed: bool = False) -> list[dict]:
    out = []
    for index, call in enumerate(calls):
        item = {
            "id": call.id,
            "type": "function",
            "function": {
                "name": call.name,
                "arguments": call.raw_arguments or json.dumps(call.arguments),
            },
        }
        if indexed:
            item["index"] = index
        out.append(item)
    return out


def completion_json(
    response: ChatResponse, messages: list[dict], completion_id: str | None = None
) -> dict:
    message: dict = {"role": "assistant", "content": response.text or None}
    if response.tool_calls:
        message["tool_calls"] = tool_calls_json(response.tool_calls)
    if response.reasoning:
        message["reasoning_content"] = response.reasoning
    finish = response.finish_reason or ("tool_calls" if response.tool_calls else "stop")
    return {
        "id": completion_id or new_id(),
        "object": "chat.completion",
        "created": int(time.time()),
        "model": response.origin.model,
        "choices": [{"index": 0, "message": message, "finish_reason": finish}],
        "usage": usage_json(estimate_usage(messages, response)),
    }


def chunk_json(
    completion_id: str,
    model: str,
    delta: Mapping,
    finish_reason: str | None = None,
    usage: Usage | None = None,
) -> dict:
    body = {
        "id": completion_id,
        "object": "chat.completion.chunk",
        "created": int(time.time()),
        "model": model,
        "choices": [{"index": 0, "delta": dict(delta), "finish_reason": finish_reason}],
    }
    if usage is not None:
        body["usage"] = usage_json(usage)
    return body


def sse(payload: Mapping | str) -> str:
    data = (
        payload if isinstance(payload, str) else json.dumps(payload, ensure_ascii=False)
    )
    return f"data: {data}\n\n"


def error_json(message: str, kind: str, code: str | None = None) -> dict:
    return {"error": {"message": message, "type": kind, "code": code}}


def models_json(names: list[str], created: int | None = None) -> dict:
    stamp = created or int(time.time())
    return {
        "object": "list",
        "data": [
            {"id": n, "object": "model", "created": stamp, "owned_by": "llm-rotator"}
            for n in names
        ],
    }
