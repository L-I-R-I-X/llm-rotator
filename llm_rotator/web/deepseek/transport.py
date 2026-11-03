"""Транспорт DeepSeek Web: ход разговора на одной связке (аккаунт + прокси).

Одна попытка = один ход: занять слот связки, решить (планировщик), писать в
существующий чат дельтой или завести новый, получить ответ и зафиксировать
состояние чата. Любой отказ превращается в `Attempt` по таблице плана (2.4),
а чат, начатый или затронутый этой попыткой, помечается брошенным.
"""

from __future__ import annotations

import json
import logging
import os
import threading
import time
from collections.abc import Iterator

from ...gateway import Attempt, Failure, Gateway
from ...registry import Binding, Route, WebModel, WebSettings
from ...transport import (
    ChatRequest,
    ChatResponse,
    Origin,
    StreamBase,
    StreamBroken,
    StreamChunk,
    ToolCall,
    Usage,
)
from .. import fingerprint as fp_module
from ..context import TransportContext
from ..errors import WebError, to_failure
from ..planner import TurnPlan, plan_turn
from ..store import ABANDONED, ACTIVE, BUSY, ChatRecord
from ..tools import (
    format_retry_prompt,
    known_call,
    looks_like_tool_call_markup,
    new_call_id,
    parse_tool_call,
)
from .account import Account, load_account
from .client import DeepSeekClient
from .sse import Delta, Event, Finish, Ready

log = logging.getLogger(__name__)

BUSY_DETAIL = "связка занята"


class _Outcome:
    """Что накопил ход: текст, рассуждение, причина остановки."""

    def __init__(self) -> None:
        self.text: list[str] = []
        self.reasoning: list[str] = []
        self.finish_reason = ""
        self.message_id: int | str | None = None
        self.tool_calls: tuple[ToolCall, ...] = ()
        self.mode = ""
        self.extra_chars = 0
        self.first_token = 0.0
        self.started = time.monotonic()

    def absorb(self, event: Event) -> None:
        if isinstance(event, Ready):
            self.message_id = event.response_message_id
        elif isinstance(event, Delta):
            if event.reasoning:
                self.reasoning.append(event.text)
            else:
                if not self.first_token:
                    self.first_token = time.monotonic() - self.started
                self.text.append(event.text)
        elif isinstance(event, Finish):
            self.finish_reason = event.reason

    def response(self, origin: Origin, attempts=()) -> ChatResponse:
        return ChatResponse(
            origin=origin,
            text="".join(self.text),
            tool_calls=self.tool_calls,
            finish_reason=self.finish_reason,
            usage=Usage(),
            latency=time.monotonic() - self.started,
            first_token=self.first_token,
            reasoning="".join(self.reasoning).strip(),
            mode=self.mode,
            attempts=attempts,
        )


class WebLiveStream(StreamBase):
    """Поток ответа веб-чата: тот же интерфейс, что у `LiveStream`."""

    def __init__(
        self,
        events: Iterator[Event],
        origin: Origin,
        outcome: _Outcome,
        pending: list[Event],
    ) -> None:
        self.origin = origin
        self.attempts = ()
        self._events = events
        self._outcome = outcome
        self._pending = pending
        self._closed = False

    @property
    def text(self) -> str:
        return "".join(self._outcome.text)

    def __iter__(self) -> Iterator[StreamChunk]:
        try:
            while self._pending:
                chunk = self._chunk(self._pending.pop(0))
                if chunk is not None:
                    yield chunk
            for event in self._events:
                self._outcome.absorb(event)
                chunk = self._chunk(event)
                if chunk is not None:
                    yield chunk
        except WebError as exc:
            self.close()
            raise StreamBroken(
                f"{self.origin.label}: поток оборвался — {exc}", self.text
            ) from exc

    @staticmethod
    def _chunk(event: Event) -> StreamChunk | None:
        if isinstance(event, Delta) and not event.reasoning:
            return StreamChunk(delta=event.text)
        if isinstance(event, Finish):
            return StreamChunk(finish_reason=event.reason)
        return None

    def collect(self) -> ChatResponse:
        return self._outcome.response(self.origin, self.attempts)

    def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        close = getattr(self._events, "close", None)
        if close:
            close()


class DeepSeekWebTransport:
    api_styles: tuple[str, ...] = ("deepseek_web",)

    def __init__(self, gateway: Gateway, context: TransportContext) -> None:
        self.gateway = gateway
        self.context = context
        self._accounts: dict[str, tuple[float, Account]] = {}
        self._lock = threading.Lock()

    # --- Transport ----------------------------------------------------------

    def complete(self, route: Route, binding: Binding, request: ChatRequest) -> Attempt:
        origin = Origin.of(route, binding)
        try:
            outcome = _Outcome()
            for event in self._turn(route, binding, request, outcome):
                outcome.absorb(event)
            return self._ok(route, binding, outcome.response(origin))
        except WebError as exc:
            return self._failed(route, binding, exc)
        except Exception as exc:  # noqa: BLE001 — не ронять обход из-за своего бага
            log.exception("DeepSeek Web: сбой хода")
            return self._attempt(
                route, binding, Failure.FATAL, detail=f"{type(exc).__name__}"
            )

    def open_stream(
        self, route: Route, binding: Binding, request: ChatRequest
    ) -> Attempt:
        origin = Origin.of(route, binding)
        outcome = _Outcome()
        events = self._turn(route, binding, request, outcome)
        pending: list[Event] = []
        try:
            # Тянем до первого куска ответа: пока клиенту ничего не отдано,
            # обход ещё может уйти на другую связку.
            for event in events:
                outcome.absorb(event)
                pending.append(event)
                if isinstance(event, Finish) or (
                    isinstance(event, Delta) and not event.reasoning
                ):
                    break
        except WebError as exc:
            return self._failed(route, binding, exc)
        except Exception as exc:  # noqa: BLE001
            log.exception("DeepSeek Web: сбой хода")
            return self._attempt(
                route, binding, Failure.FATAL, detail=f"{type(exc).__name__}"
            )
        # `pending` уже учтён в outcome; WebLiveStream учтёт только остальное.
        return self._ok(route, binding, WebLiveStream(events, origin, outcome, pending))

    def close(self) -> None:
        self._accounts.clear()

    # --- ход ----------------------------------------------------------------

    def _settings(self, route: Route) -> WebSettings:
        web = route.provider.web
        assert web is not None
        return web

    def _account(self, path: str) -> Account:
        mtime = os.stat(path).st_mtime if os.path.exists(path) else 0.0
        with self._lock:
            cached = self._accounts.get(path)
            if cached and cached[0] == mtime:
                return cached[1]
        account = load_account(path)
        with self._lock:
            self._accounts[path] = (mtime, account)
        return account

    def _turn(
        self, route: Route, binding: Binding, request: ChatRequest, outcome: _Outcome
    ) -> Iterator[Event]:
        """Генератор хода. Освобождает слот и разбирается с чатом в `finally`."""
        context = self.context
        key = binding.key_name
        if not context.busy.acquire(key, binding.max_concurrent):
            raise WebError("busy_slot", BUSY_DETAIL)
        record: ChatRecord | None = None
        committed = False
        try:
            settings = self._settings(route)
            tools = self._tools(request)
            hint = request.conversation
            namespace = hint.namespace if hint else "default"
            session = hint.session if hint else ""
            account = self._account(binding.key_value)
            solver = {"pow_solver": context.pow_solver} if context.pow_solver else {}
            client = DeepSeekClient(
                account,
                context.http(binding.proxy),
                settings,
                route.provider.base_url,
                **solver,
            )
            flags = settings.model(route.model.name) or WebModel()
            plan = plan_turn(
                context.store,
                settings,
                provider=route.provider.name,
                key_name=key,
                namespace=namespace,
                session=session,
                messages=request.messages,
                tools=tools,
            )
            record = plan.chat
            outcome.mode = "file" if plan.attachments else plan.kind
            if plan.too_large:
                raise WebError("too_large", plan.too_large)
            refs = self._upload(client, key, plan.attachments)

            def fresh() -> ChatRecord:
                chat_id = client.create_chat()
                return context.store.create_chat(
                    route.provider.name,
                    chat_id,
                    key,
                    binding.proxy.name,
                    namespace,
                    session,
                    plan.fingerprint.context_key,
                    BUSY,
                )

            if plan.kind == "full":
                record = fresh()
                prompt, parent = plan.prompt, None
            else:
                assert record is not None
                prompt, parent = plan.prompt, record.parent_message_id
            events = client.complete(record.chat_id, parent, prompt, refs, flags=flags)
            try:
                first = next(events)
            except WebError as exc:
                if exc.kind != "chat_gone" or plan.kind != "delta":
                    raise
                # Чат пропал на стороне сайта: новый чат, полный контекст,
                # в той же попытке.
                context.store.set_state(route.provider.name, record.chat_id, ABANDONED)
                record = fresh()
                if plan.full_too_large():
                    raise WebError("too_large", plan.full_reason)
                refs = self._upload(client, key, plan.full_attachments)
                events = client.complete(
                    record.chat_id, None, plan.full_prompt, refs, flags=flags
                )
                prompt = plan.full_prompt
                outcome.mode = "file" if plan.full_attachments else "full"
                first = next(events)
            if tools:
                # Вызов инструмента нельзя отдавать по кускам: сначала ответ
                # целиком, потом разбор (и при необходимости уточнение).
                yield from self._tool_turn(
                    client, record, first, events, flags, tools, outcome
                )
            else:
                yield first
                yield from events
            if not outcome.text and not outcome.tool_calls:
                raise WebError("empty", "ответ без текста")
            self._commit(route, plan, record, prompt, outcome, settings)
            committed = True
            context.store.mark_account(key, "ok")
        finally:
            if record is not None and not committed:
                context.store.set_state(route.provider.name, record.chat_id, ABANDONED)
            context.busy.release(key)
            if record is not None:
                context.wake()  # брошенное (или after_turn) — пусть уберут

    @staticmethod
    def _tools(request: ChatRequest) -> object:
        """Инструменты запроса; `tool_choice="none"` — как будто их нет."""
        if request.params.get("tool_choice") == "none":
            return None
        return request.params.get("tools") or None

    @staticmethod
    def _drain(first: Event, events: Iterator[Event]) -> _Outcome:
        local = _Outcome()
        local.absorb(first)
        for event in events:
            local.absorb(event)
        return local

    def _tool_turn(
        self, client, record, first, events, flags, tools, outcome: _Outcome
    ) -> Iterator[Event]:
        local = self._drain(first, events)
        text = "".join(local.text)
        call = known_call(parse_tool_call(text), tools)
        if call is None and looks_like_tool_call_markup(text):
            # Разметка битая. Просим повторить В ТОМ ЖЕ чате: новый чат ради
            # этого был бы ещё одним следом в аккаунте (R1).
            retry_prompt = format_retry_prompt()
            stream = client.complete(
                record.chat_id, local.message_id, retry_prompt, flags=flags
            )
            again = self._drain(next(stream), stream)
            again.extra_chars = local.extra_chars + len(retry_prompt) + len(text)
            local = again
            text = "".join(local.text)
            call = known_call(parse_tool_call(text), tools)
        outcome.message_id = local.message_id
        outcome.reasoning = local.reasoning
        outcome.extra_chars = local.extra_chars
        if call is not None:
            outcome.tool_calls = (
                ToolCall(
                    id=new_call_id(),
                    name=call.name,
                    arguments=call.arguments,
                    raw_arguments=json.dumps(call.arguments, ensure_ascii=False),
                ),
            )
            outcome.finish_reason = "tool_calls"
            yield Finish("tool_calls")
            return
        yield Delta(text)
        yield Finish(local.finish_reason or "stop")

    def _commit(
        self,
        route: Route,
        plan: TurnPlan,
        record: ChatRecord,
        prompt: str,
        outcome: _Outcome,
        settings: WebSettings,
    ) -> None:
        text = "".join(outcome.text)
        fp = plan.fingerprint
        calls = [(c.name, dict(c.arguments)) for c in outcome.tool_calls]
        sent = (*fp.request_keys, fp_module.reply_key(text, calls))
        added = (
            len(prompt)
            + len(text)
            + len("".join(outcome.reasoning))
            + outcome.extra_chars
        )
        provider = route.provider.name
        store = self.context.store
        store.finish_turn(
            provider,
            record.chat_id,
            sent,
            outcome.message_id,
            added,
            ABANDONED
            if plan.ephemeral or settings.delete_policy == "after_turn"
            else ACTIVE,
        )
        if not plan.ephemeral:
            hint_ns, hint_session = record.namespace, record.session
            store.abandon_conversation(
                provider, hint_ns, hint_session, fp.first_key, record.chat_id
            )

    def _upload(self, client, key: str, attachments) -> list[str]:
        """Файлы хода — на аккаунт связки. Одинаковое содержимое грузим один раз."""
        store = self.context.store
        ids: list[str] = []
        for attachment in attachments:
            digest = attachment.sha256
            file_id = store.cached_file(key, digest)
            if file_id is None:
                file_id = client.upload_file(
                    attachment.name, attachment.data, attachment.mime
                )
                store.remember_file(key, digest, file_id)
            ids.append(file_id)
        return ids

    # --- Attempt ------------------------------------------------------------

    def _ok(self, route: Route, binding: Binding, payload: object) -> Attempt:
        return Attempt(
            model=route.model.name,
            provider=route.provider.name,
            proxy=binding.proxy,
            key_name=binding.key_name,
            failure=Failure.NONE,
            status=200,
            payload=payload,
        )

    def _attempt(
        self,
        route: Route,
        binding: Binding,
        failure: Failure,
        status: int | None = None,
        detail: str = "",
        retry_after: float | None = None,
        transient: bool = False,
    ) -> Attempt:
        return Attempt(
            model=route.model.name,
            provider=route.provider.name,
            proxy=binding.proxy,
            key_name=binding.key_name,
            failure=failure,
            status=status,
            detail=detail,
            retry_after=retry_after,
            transient=transient,
        )

    def _failed(self, route: Route, binding: Binding, error: WebError) -> Attempt:
        if error.kind == "busy_slot":
            # Не отказ аккаунта: он просто занят. Идём дальше без штрафа.
            return self._attempt(
                route,
                binding,
                Failure.QUOTA,
                detail=BUSY_DETAIL,
                retry_after=0.0,
                transient=True,
            )
        failure, retry_after = to_failure(error)
        if failure is Failure.PROXY and binding.proxy.is_direct:
            # Молчит аккаунт или сайт, а не «адрес»: прямое соединение общее
            # для всех аккаунтов, и штрафовать его значило бы отрезать и те,
            # что работают. Ставим на паузу только эту связку.
            failure, retry_after = Failure.QUOTA, 30.0
        provider = route.provider.name
        if error.kind == "auth":
            self.context.store.mark_account(
                binding.key_name, "needs_auth", error.detail
            )
        elif error.kind == "busy":
            self.gateway.state.note_busy(provider, binding.key_name)
        log.debug(
            "%s через %s ключ %s → %s %s",
            route.model.name,
            binding.proxy.name,
            binding.key_name,
            failure.value,
            error.kind,
        )
        return self._attempt(
            route,
            binding,
            failure,
            status=error.status,
            detail=str(error),
            retry_after=retry_after,
        )
