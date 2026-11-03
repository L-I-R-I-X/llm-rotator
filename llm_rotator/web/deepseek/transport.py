"""Транспорт DeepSeek Web: ход разговора на одной связке (аккаунт + прокси).

Одна попытка = один ход: занять слот связки, решить (планировщик), писать в
существующий чат дельтой или завести новый, получить ответ и зафиксировать
состояние чата. Любой отказ превращается в `Attempt` по таблице плана (2.4),
а чат, начатый или затронутый этой попыткой, помечается брошенным.
"""

from __future__ import annotations

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
    Usage,
)
from .. import fingerprint as fp_module
from ..context import TransportContext
from ..errors import WebError, to_failure
from ..planner import TurnPlan, plan_turn
from ..store import ABANDONED, ACTIVE, BUSY, ChatRecord
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
            finish_reason=self.finish_reason,
            usage=Usage(),
            latency=time.monotonic() - self.started,
            first_token=self.first_token,
            reasoning="".join(self.reasoning).strip(),
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
            self._reject_unsupported(request)
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
                tools=request.params.get("tools"),
            )
            record = plan.chat
            self._check_size(plan, settings)

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
            events = client.complete(record.chat_id, parent, prompt, flags=flags)
            try:
                first = next(events)
            except WebError as exc:
                if exc.kind != "chat_gone" or plan.kind != "delta":
                    raise
                # Чат пропал на стороне сайта: новый чат, полный контекст,
                # в той же попытке.
                context.store.set_state(route.provider.name, record.chat_id, ABANDONED)
                record = fresh()
                events = client.complete(
                    record.chat_id, None, plan.full_prompt, flags=flags
                )
                prompt = plan.full_prompt
                first = next(events)
            yield first
            yield from events
            if not outcome.text:
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
        sent = (*fp.request_keys, fp_module.reply_key(text))
        added = len(prompt) + len(text) + len("".join(outcome.reasoning))
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

    @staticmethod
    def _reject_unsupported(request: ChatRequest) -> None:
        if request.params.get("tools"):
            raise WebError(
                "bad_request", "tool calling для веб-чатов ещё не реализован (этап 5)"
            )
        if any(fp_module.has_images(m) for m in request.messages):
            raise WebError("bad_request", "картинки для веб-чатов ещё не реализованы")

    @staticmethod
    def _check_size(plan: TurnPlan, settings: WebSettings) -> None:
        if len(plan.prompt) > settings.inline_max_chars:
            raise WebError(
                "too_large",
                f"контекст {len(plan.prompt)} символов больше inline_max_chars "
                f"({settings.inline_max_chars}); отправка файлом — этап 6",
            )

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
        )

    def _failed(self, route: Route, binding: Binding, error: WebError) -> Attempt:
        if error.kind == "busy_slot":
            # Не отказ аккаунта: он просто занят. Идём дальше без штрафа.
            return self._attempt(
                route, binding, Failure.QUOTA, detail=BUSY_DETAIL, retry_after=0.0
            )
        failure, retry_after = to_failure(error)
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
