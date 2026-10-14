"""Отправка запросов к моделям через openai-совместимый слой.

Транспорт не занимается обходом: за провайдеров, ключи и адреса отвечает
Gateway.rotate, а транспорт делает ровно одну попытку на связке (ключ, адрес)
и честно говорит, чем она кончилась. Возвращается Attempt, у которого при
Failure.NONE в payload лежит ChatResponse или LiveStream.

Один класс на все OpenAI-совместимые API — Groq, Google, OpenRouter, Ollama,
vLLM, LM Studio, Together, — потому что у них один и тот же формат тела и
ответа. Отличаются только адрес (Provider.base_url), заголовки
(Provider.extra_headers) и то, нужен ли ключ вообще (Provider.auth = "none").
Своя разводка на провайдера здесь была бы копипастой без пользы.

Почему openai, а не сборка тела руками: типы и разбор ответов уже написаны и
покрывают крайние случаи провайдеров, а главное — у библиотеки есть
`http_client`, в который кладётся наш per-proxy httpx.Client. Так ключ ходит
через свой адрес, и обход адресов остаётся в шлюзе, а не внутри библиотеки.

`max_retries=0` обязателен: по умолчанию библиотека сама повторяет запрос два
раза, и мы бы не увидели, сколько на самом деле попыток, а сломанный ключ
пять раз подряд подряд ушёл бы в бан по нашей вине.

Ошибки SDK разбираются в Failure теми же правилами, что и в шлюзе: у
openai.APIStatusError в .response лежит тот самый httpx.Response, поэтому
работает готовый classify(), а у обрыва и таймаута (APIConnectionError)
ответа нет вовсе — это сломанный адрес.

Сырые запросы для несовместимых API (нативный Gemini, Anthropic) в
llm_transport не делаются: под них в TRANSPORTS добавляется свой класс с
той же сигнатурой, а выбор идёт по Provider.api_style.
"""

from __future__ import annotations

import json
import logging
import threading
import time
from collections.abc import Callable, Iterator, Mapping
from dataclasses import dataclass, field
from typing import Protocol, Self

import openai
from httpx import HTTPError

from .gateway import (
    Attempt,
    Failure,
    Gateway,
    LLMError,
    classify,
    error_detail,
    retry_after,
)
from .registry import Binding, ConfigError, Provider, Route

log = logging.getLogger(__name__)

STREAM_ERRORS = (LLMError, openai.OpenAIError, HTTPError)
"""Что может оборвать поток на середине. Ротация тут уже невозможна."""

LOCAL_API_KEY = "local"
"""Заглушка вместо ключа для Provider.auth = "none".

Библиотека требует непустой api_key и всегда шлёт Authorization: Bearer, а
провайдер без авторизации этот заголовок либо игнорирует, либо не требует.
Придумывать настоящий ключ нельзя — его неотличимый от настоящего мусор
попадёт в логи и в UI.
"""


class StreamBroken(LLMError):
    """Поток оборвался уже после начала.

    Ротация тут бессильна: начало ответа показано, и повтор запроса на другом
    ключе продублировал бы текст. Поэтому это отдельная ошибка, а не возврат
    к обходу. Частичный текст остаётся в partial_text.
    """

    def __init__(self, detail: str, partial_text: str = "") -> None:
        super().__init__(detail)
        self.detail = detail
        self.partial_text = partial_text


@dataclass(frozen=True, slots=True)
class Origin:
    """Кто написал конкретное сообщение.

    Имя переменной .env, а не значение: значение ключа не покидает
    process.env и ни одного лога, а различать ключи для отладки и так
    достаточно по именам.
    """

    model: str
    provider: str
    key_name: str
    proxy_name: str

    @classmethod
    def of(cls, route: Route, binding: Binding) -> Origin:
        return cls(
            model=route.model.name,
            provider=route.provider.name,
            key_name=binding.key_name,
            proxy_name=binding.proxy.name,
        )

    @property
    def label(self) -> str:
        return f"{self.model} @ {self.provider}"

    def line(self) -> str:
        """Подпись под сообщением в чате: всё, по чему можно опознать автора."""
        return f"{self.model} · {self.provider} · {self.key_name} · {self.proxy_name}"

    def __str__(self) -> str:
        return self.line()


@dataclass(frozen=True, slots=True)
class Usage:
    """Расход токенов. Нули у провайдеров, которые не присылают usage."""

    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_tokens: int = 0
    cached_tokens: int = 0

    def __str__(self) -> str:
        return f"{self.prompt_tokens}+{self.completion_tokens}={self.total_tokens}"


@dataclass(frozen=True, slots=True)
class ToolCall:
    """Вызов инструмента, который модель попросила.

    `arguments` — распарсенный JSON, `raw_arguments` — как прислали. Битый JSON
    не роняет вызов: остаётся в raw_arguments, чтобы вызывающий увидел, что
    модель выдала, и не выдавал пустые аргументы за «инструмент не хотел
    ничего».
    """

    id: str
    name: str
    arguments: Mapping[str, object] = field(default_factory=dict)
    raw_arguments: str = ""


@dataclass(frozen=True, slots=True)
class ChatRequest:
    """Запрос к модели, общий для всех провайдеров маршрута.

    `params` — всё, что провайдеру сверх model и messages: temperature,
    max_tokens, tools, tool_choice, response_format, seed, и
    `extra_body={"top_k": ...}` для того, чего нет в сигнатуре библиотеки.
    Ключи, которые библиотека не знает, в params класть нельзя: она требует
    extra_body. Параметры, которые не понимает конкретный провайдер, приедут к
    нему как 400 и обход не поможет.
    """

    model: str
    messages: list[dict]
    params: Mapping[str, object] = field(default_factory=dict)
    read_timeout: float | None = None

    def as_body(self) -> dict:
        return {
            "model": self.model,
            "messages": self.messages,
            **dict(self.params),
        }

    def with_(self, **overrides: object) -> dict:
        """Тело запроса с добавленными полями вызова (stream, timeout)."""
        return {**self.as_body(), **overrides}


@dataclass(frozen=True, slots=True)
class ChatResponse:
    """Ответ модели вместе с тем, кто и откуда его принёс."""

    origin: Origin
    text: str = ""
    tool_calls: tuple[ToolCall, ...] = ()
    finish_reason: str = ""
    usage: Usage = Usage()
    latency: float = 0.0
    reasoning: str = ""
    """Ход рассуждения, если провайдер его присылает (нестандартное поле у
    Groq и части локальных серверов). В тело следующего запроса не идёт."""
    attempts: tuple[Attempt, ...] = ()
    """Неудачные попытки обхода до успешной: из них получается сноска в чате."""
    raw: object = field(default=None, repr=False, compare=False)

    def as_message(self) -> dict:
        """Сообщение ассистента в формате провайдера, для следующего хода."""
        message: dict = {"role": "assistant", "content": self.text or None}
        if self.tool_calls:
            message["tool_calls"] = [
                {
                    "id": call.id,
                    "type": "function",
                    "function": {
                        "name": call.name,
                        "arguments": call.raw_arguments or json.dumps(call.arguments),
                    },
                }
                for call in self.tool_calls
            ]
        return message


@dataclass(frozen=True, slots=True)
class StreamChunk:
    """Одна порция ответа при стриминге."""

    delta: str = ""
    finish_reason: str = ""
    usage: Usage | None = None


def failover_notes(attempts: tuple[Attempt, ...]) -> tuple[str, ...]:
    """Сноски о том, кого перебирали до успеха. Пусто, если с первого раза."""
    return tuple(
        f"сначала: {attempt}"
        for attempt in attempts
        if attempt.failure is not Failure.NONE
    )


class Transport(Protocol):
    """Одна попытка на связке (ключ, адрес). Обход — дело Gateway.rotate."""

    def complete(
        self, route: Route, binding: Binding, request: ChatRequest
    ) -> Attempt: ...

    def open_stream(
        self, route: Route, binding: Binding, request: ChatRequest
    ) -> Attempt: ...

    def close(self) -> None: ...


class _ToolCallBuilder:
    """Склейка фрагментов tool_call из стрима.

    Провайдер шлёт id и имя только в первой дельте вызова, а аргументы
    приходят кусками. Склеивать надо по индексу, а не по порядку прихода.
    """

    def __init__(self) -> None:
        self._entries: dict[int, dict[str, str]] = {}

    def add(self, deltas: object) -> None:
        for delta in deltas or ():
            entry = self._entries.setdefault(delta.index, {"id": "", "name": ""})
            if delta.id:
                entry["id"] = delta.id
            function = delta.function
            if function is None:
                continue
            if function.name:
                entry["name"] = function.name
            if function.arguments:
                entry["arguments"] = entry.get("arguments", "") + function.arguments

    def build(self) -> tuple[ToolCall, ...]:
        return tuple(
            _tool_call(
                entry.get("id", ""),
                entry.get("name", ""),
                entry.get("arguments", ""),
            )
            for _, entry in sorted(self._entries.items())
        )


def _tool_call(call_id: str, name: str, raw: str) -> ToolCall:
    try:
        parsed = json.loads(raw) if raw.strip() else {}
    except ValueError:
        log.warning("Tool call %s: аргументы не JSON, беру как есть", name)
        parsed = {}
    if not isinstance(parsed, dict):
        log.warning("Tool call %s: аргументы — не объект JSON", name)
        parsed = {}
    return ToolCall(id=call_id, name=name, arguments=parsed, raw_arguments=raw)


class LiveStream:
    """Живой поток ответа: итерация по дельтам плюс накопленное.

    Первый чанк уже получен — open_stream() вытягивает его, чтобы обход
    успел найти живую связку до того, как наружу поедет хоть один символ. Он не
    теряется: отдаётся первым при итерации.
    """

    def __init__(
        self,
        stream: object,
        origin: Origin,
        started: float,
        first: object = None,
        on_close: Callable[[LiveStream], None] | None = None,
    ) -> None:
        self.origin = origin
        self.attempts: tuple[Attempt, ...] = ()
        self._stream = stream
        self._started = started
        self._first = first
        self._on_close = on_close
        self._text: list[str] = []
        self._tools = _ToolCallBuilder()
        self._finish_reason = ""
        self._usage: Usage | None = None
        self._closed = False

    @property
    def text(self) -> str:
        """Всё, что успело прийти, включая недобранный хвост."""
        return "".join(self._text)

    def __iter__(self) -> Iterator[StreamChunk]:
        try:
            if self._first is not None:
                first, self._first = self._first, None
                chunk = self._absorb(first)
                if chunk is not None:
                    yield chunk
            for raw in self._stream:
                chunk = self._absorb(raw)
                if chunk is not None:
                    yield chunk
        except STREAM_ERRORS as exc:
            self.close()
            raise StreamBroken(
                f"{self.origin.label}: поток оборвался — {type(exc).__name__}: {exc}",
                self.text,
            ) from exc

    def _absorb(self, raw: object) -> StreamChunk | None:
        """Принять чанк провайдера: накопить и отдать дельту."""
        choices = getattr(raw, "choices", None) or ()
        usage = _usage(getattr(raw, "usage", None))
        if usage is not None:
            self._usage = usage
        if not choices:
            return StreamChunk(usage=usage) if usage is not None else None
        choice = choices[0]
        delta = getattr(choice, "delta", None)
        if delta is not None:
            self._tools.add(getattr(delta, "tool_calls", None))
        if choice.finish_reason:
            self._finish_reason = choice.finish_reason
        text = getattr(delta, "content", None) or ""
        if text:
            self._text.append(text)
        return StreamChunk(
            delta=text,
            finish_reason=choice.finish_reason or "",
            usage=usage,
        )

    def collect(self) -> ChatResponse:
        """Ответ целиком из того, что накоплено. Работает и на недочитанном."""
        return ChatResponse(
            origin=self.origin,
            text=self.text,
            tool_calls=self._tools.build(),
            finish_reason=self._finish_reason,
            usage=self._usage or Usage(),
            latency=time.monotonic() - self._started,
            attempts=self.attempts,
        )

    def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        try:
            self._stream.close()
        except Exception:
            log.debug("Поток %s не закрылся", self.origin.label, exc_info=True)
        if self._on_close is not None:
            self._on_close(self)

    def __enter__(self) -> Self:
        return self

    def __exit__(self, *exc_info: object) -> None:
        self.close()


class OpenAIChatTransport:
    """OpenAI-совместимый API: Groq, Google, OpenRouter, Ollama, vLLM, …"""

    api_styles: tuple[str, ...] = ("openai",)

    def __init__(self, gateway: Gateway) -> None:
        self.gateway = gateway
        self._clients: dict[tuple[str, str, str, str], openai.OpenAI] = {}
        self._streams: set[LiveStream] = set()
        self._lock = threading.Lock()

    def client(self, provider: Provider, binding: Binding) -> openai.OpenAI:
        """Клиент библиотеки под конкретную связку (адрес, ключ).

        Кэшируется, потому что в нём держится пул соединений httpx: клиент на
        каждый запрос означал бы новый TLS-хендшейк. Ключ кэша включает адрес
        и способ авторизации, иначе две связки одного провайдера поделили бы
        клиент и ушли бы не туда.
        """
        cache_key = (
            provider.base_url,
            provider.auth,
            binding.key_name,
            binding.proxy.name,
        )
        client = self._clients.get(cache_key)
        if client is not None:
            return client
        with self._lock:
            client = self._clients.get(cache_key)
            if client is None:
                client = openai.OpenAI(
                    api_key=binding.key_value or LOCAL_API_KEY,
                    base_url=provider.base_url,
                    http_client=self.gateway.client(binding.proxy),
                    max_retries=0,
                    default_headers=provider.headers,
                )
                self._clients[cache_key] = client
        return client

    def complete(self, route: Route, binding: Binding, request: ChatRequest) -> Attempt:
        origin = Origin.of(route, binding)
        if request.params.get("stream"):
            return self._failed(
                route,
                binding,
                Failure.FATAL,
                detail="stream=True только через stream()",
            )
        call: dict = {"stream": False}
        if request.read_timeout is not None:
            call["timeout"] = request.read_timeout
        started = time.monotonic()
        try:
            completion = self.client(route.provider, binding).chat.completions.create(
                **request.with_(**call)
            )
        except openai.APIStatusError as exc:
            return self._failed(
                route,
                binding,
                classify(exc.response),
                status=exc.status_code,
                detail=error_detail(exc.response),
                retry_after=retry_after(exc.response),
            )
        except openai.APIConnectionError as exc:
            return self._failed(
                route, binding, Failure.PROXY, detail=type(exc).__name__
            )
        except openai.OpenAIError as exc:
            return self._failed(
                route, binding, Failure.FATAL, detail=f"{type(exc).__name__}: {exc}"
            )

        choice = completion.choices[0] if completion.choices else None
        message = choice.message if choice is not None else None
        return Attempt(
            model=route.model.name,
            provider=route.provider.name,
            proxy=binding.proxy,
            key_name=binding.key_name,
            failure=Failure.NONE,
            status=200,
            payload=ChatResponse(
                origin=origin,
                text=getattr(message, "content", None) or "",
                tool_calls=tuple(
                    _tool_call(
                        call.id, call.function.name, call.function.arguments or ""
                    )
                    for call in (getattr(message, "tool_calls", None) or ())
                ),
                finish_reason=(choice.finish_reason or "") if choice else "",
                usage=_usage(completion.usage) or Usage(),
                latency=time.monotonic() - started,
                reasoning=_reasoning(message),
                raw=completion,
            ),
        )

    def open_stream(
        self, route: Route, binding: Binding, request: ChatRequest
    ) -> Attempt:
        origin = Origin.of(route, binding)
        call: dict = {"stream": True}
        if request.read_timeout is not None:
            call["timeout"] = request.read_timeout
        started = time.monotonic()
        try:
            raw_stream = self.client(route.provider, binding).chat.completions.create(
                **request.with_(**call)
            )
            first = next(iter(raw_stream), None)
        except openai.APIStatusError as exc:
            return self._failed(
                route,
                binding,
                classify(exc.response),
                status=exc.status_code,
                detail=error_detail(exc.response),
                retry_after=retry_after(exc.response),
            )
        except openai.APIConnectionError as exc:
            return self._failed(
                route, binding, Failure.PROXY, detail=type(exc).__name__
            )
        except openai.OpenAIError as exc:
            return self._failed(
                route, binding, Failure.FATAL, detail=f"{type(exc).__name__}: {exc}"
            )

        if first is None:
            raw_stream.close()
            return self._failed(
                route,
                binding,
                Failure.FATAL,
                detail="провайдер закрыл поток, не отдав ни одного чанка",
            )
        live = LiveStream(
            raw_stream,
            origin=origin,
            started=started,
            first=first,
            on_close=self._forget,
        )
        with self._lock:
            self._streams.add(live)
        return Attempt(
            model=route.model.name,
            provider=route.provider.name,
            proxy=binding.proxy,
            key_name=binding.key_name,
            failure=Failure.NONE,
            status=200,
            payload=live,
        )

    def close(self) -> None:
        with self._lock:
            streams = tuple(self._streams)
        for stream in streams:
            stream.close()
        self._clients.clear()

    def _forget(self, stream: LiveStream) -> None:
        with self._lock:
            self._streams.discard(stream)

    def _failed(
        self,
        route: Route,
        binding: Binding,
        failure: Failure,
        status: int | None = None,
        detail: str = "",
        retry_after: float | None = None,
    ) -> Attempt:
        log.debug(
            "%s через %s ключ %s → %s %s",
            route.model.name,
            binding.proxy.name,
            binding.key_name,
            failure.value,
            detail,
        )
        return Attempt(
            model=route.model.name,
            provider=route.provider.name,
            proxy=binding.proxy,
            key_name=binding.key_name,
            failure=failure,
            status=status,
            detail=detail,
            retry_after=retry_after,
        )


def _usage(raw: object) -> Usage | None:
    if raw is None:
        return None
    details = getattr(raw, "prompt_tokens_details", None)
    return Usage(
        prompt_tokens=getattr(raw, "prompt_tokens", 0) or 0,
        completion_tokens=getattr(raw, "completion_tokens", 0) or 0,
        total_tokens=getattr(raw, "total_tokens", 0) or 0,
        cached_tokens=(getattr(details, "cached_tokens", 0) or 0) if details else 0,
    )


def _reasoning(message: object) -> str:
    """Ход рассуждения у провайдеров, которые его присылают нестандартным полем."""
    for field_name in ("reasoning", "reasoning_content"):
        value = getattr(message, field_name, None)
        if isinstance(value, str) and value.strip():
            return value.strip()
    return ""


TransportFactory = Callable[[Gateway], Transport]

TRANSPORTS: dict[str, TransportFactory] = {
    "openai": OpenAIChatTransport,
}
"""Способы отправки по Provider.api_style.

Реестр независим от llm_gateway.API_STYLES: там заголовки авторизации и
разбор списка моделей, здесь — как уходит запрос к самой модели. Под
несовместимый API (нативный Gemini, Anthropic) добавляется класс с той же
сигнатурой Transport и строка в этом словаре.
"""


def make_transport(provider: Provider, gateway: Gateway) -> Transport:
    try:
        factory = TRANSPORTS[provider.api_style]
    except KeyError:
        raise ConfigError(
            f"Провайдер {provider.name!r}: отправлять запросы в стиле "
            f"{provider.api_style!r} нечем. Умею: {_styles()}"
        ) from None
    return factory(gateway)


def _styles() -> str:
    return ", ".join(repr(s) for s in sorted(TRANSPORTS)) if TRANSPORTS else "(пусто)"


__all__ = [
    "LOCAL_API_KEY",
    "TRANSPORTS",
    "ChatRequest",
    "ChatResponse",
    "LiveStream",
    "OpenAIChatTransport",
    "Origin",
    "StreamBroken",
    "StreamChunk",
    "ToolCall",
    "Transport",
    "Usage",
    "failover_notes",
    "make_transport",
]
