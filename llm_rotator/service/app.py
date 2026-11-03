"""HTTP-приложение сервиса: OpenAI-совместимые эндпоинты поверх `ModelRunner`."""

from __future__ import annotations

import asyncio
import hmac
import logging
import os
import time
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from pathlib import Path

try:
    from starlette.applications import Starlette
    from starlette.concurrency import run_in_threadpool
    from starlette.requests import Request
    from starlette.responses import JSONResponse, Response, StreamingResponse
    from starlette.routing import Route
except ImportError as exc:  # pragma: no cover
    raise ImportError(
        "для режима сервиса нужны starlette и uvicorn: pip install 'llm-rotator[server]'"
    ) from exc

from ..gateway import Failure, GatewayError
from ..registry import ConfigError, load_env_keys, load_registry
from ..runner import ModelRunner
from ..transport import ChatResponse, ConversationHint, StreamBase, StreamBroken
from . import persist, status
from .chains import ChainRunner
from .clients import ClientTable, load_clients
from .openai_api import (
    RequestError,
    chunk_json,
    completion_json,
    error_json,
    estimate_usage,
    models_json,
    new_id,
    parse_chat_request,
    sse,
    tool_calls_json,
)

log = logging.getLogger("llm_rotator.service")

KEEPALIVE = 15.0
PERSIST_EVERY = 60.0
ADMIN_TOKEN_VAR = "LLM_ROTATOR_ADMIN_TOKEN"
LOOPBACK = frozenset({"127.0.0.1", "::1", "localhost"})


class Service:
    """Один процесс: один `ModelRunner`, один `SessionState` на всё время жизни."""

    def __init__(self, base_dir: Path, runner: ModelRunner | None = None) -> None:
        self.base_dir = base_dir
        registry = load_registry(base_dir) if runner is None else runner.registry
        self.runner = runner or ModelRunner(registry)
        self.chains = ChainRunner(self.runner)
        self.env = _env(base_dir)
        self.clients: ClientTable = load_clients(base_dir, self.env, registry)
        self.admin_token = self.env.get(ADMIN_TOKEN_VAR) or os.environ.get(
            ADMIN_TOKEN_VAR, ""
        )
        if any(p.web is not None for p in registry.providers.values()):
            self.runner._web_context()  # хранилище нужно и для сохранения пауз

    def reload(self) -> None:
        """Перечитать конфигурацию без рестарта; паузы и аффинити остаются."""
        registry = load_registry(self.base_dir)
        self.env = _env(self.base_dir)
        clients = load_clients(self.base_dir, self.env, registry)
        self.runner.reload(registry)
        self.clients = clients

    @property
    def store(self):
        context = self.runner._context
        return context.store if context is not None else None

    def close(self) -> None:
        self.runner.close()


def _env(base_dir: Path) -> dict[str, str]:
    try:
        return load_env_keys(base_dir / ".env")
    except ConfigError:
        return {}


# --- ответы --------------------------------------------------------------------


def _error(status_code: int, message: str, kind: str, headers: dict | None = None):
    return JSONResponse(
        error_json(message, kind), status_code=status_code, headers=headers or {}
    )


def gateway_error_response(error: GatewayError, service: Service) -> Response:
    """503 — все пробовали и не вышло; 429 — всё на паузе по квоте; 400 — запрос плох."""
    attempts = error.attempts
    message = str(error)
    if any(a.failure is Failure.FATAL for a in attempts):
        fatal = next(a for a in attempts if a.failure is Failure.FATAL)
        return _error(
            400, f"{message}\n{fatal.detail}".strip(), "invalid_request_error"
        )
    if attempts and all(a.failure is Failure.QUOTA for a in attempts):
        wait = service.runner.gateway.state.waiting()
        if wait > 0:
            return _error(
                429, message, "rate_limit_error", {"Retry-After": str(int(wait) + 1)}
            )
    return _error(503, message, "service_unavailable")


def _headers(response: ChatResponse | StreamBase) -> dict[str, str]:
    origin = response.origin.line().encode("ascii", "replace").decode()
    headers = {
        "X-Rotator-Origin": origin,
        "X-Rotator-Attempts": str(
            len([a for a in response.attempts if a.failure is not Failure.NONE])
        ),
    }
    return headers


def _log_turn(client: str, models, response: ChatResponse, started: float) -> None:
    log.info(
        "turn client=%s model=%s origin=%s attempts=%d ttft=%.2fs total=%.2fs mode=%s",
        client,
        "→".join(models),
        response.origin.line(),
        len(response.attempts),
        response.first_token,
        time.monotonic() - started,
        response.mode or "-",
    )


# --- приложение ----------------------------------------------------------------


def create_app(service: Service) -> Starlette:
    def authenticate(request: Request):
        """Клиент по токену или готовый ответ с ошибкой."""
        table = service.clients
        if not table.requires_token:
            host = request.client.host if request.client else ""
            if host not in LOOPBACK:
                return _error(
                    401,
                    "clients.toml не настроен: без токенов сервис отвечает только "
                    "с loopback-адреса",
                    "authentication_error",
                )
            return table.open_client
        token = ""
        header = request.headers.get("authorization", "")
        if header.lower().startswith("bearer "):
            token = header[7:].strip()
        token = token or request.headers.get("x-api-key", "").strip()
        client = table.authenticate(token) if token else None
        if client is None:
            return _error(
                401, "неизвестный или отсутствующий токен", "authentication_error"
            )
        return client

    async def health(request: Request) -> Response:
        return JSONResponse({"status": "ok"})

    async def list_models(request: Request) -> Response:
        client = authenticate(request)
        if isinstance(client, Response):
            return client
        registry = service.runner.registry
        names = [m.name for m in registry.enabled_models()]
        names += [f"chain:{name}" for name in registry.chains]
        return JSONResponse(models_json(names))

    async def get_status(request: Request) -> Response:
        client = authenticate(request)
        if isinstance(client, Response):
            return client
        data = await run_in_threadpool(status.snapshot, service.runner)
        data["clients"] = list(service.clients.names)
        return JSONResponse(data)

    async def reload(request: Request) -> Response:
        host = request.client.host if request.client else ""
        token = request.headers.get("authorization", "").removeprefix("Bearer ").strip()
        allowed = host in LOOPBACK or (
            service.admin_token
            and hmac.compare_digest(token.encode(), service.admin_token.encode())
        )
        if not allowed:
            return _error(
                403,
                "перечитать конфигурацию можно только с loopback",
                "permission_error",
            )
        try:
            await run_in_threadpool(service.reload)
        except ConfigError as exc:
            return _error(
                400,
                f"конфигурация не принята, работаю на прежней: {exc}",
                "invalid_request_error",
            )
        return JSONResponse({"status": "reloaded"})

    async def chat(request: Request) -> Response:
        client = authenticate(request)
        if isinstance(client, Response):
            return client
        try:
            body = await request.json()
        except ValueError:
            return _error(400, "тело запроса не JSON", "invalid_request_error")
        try:
            parsed = parse_chat_request(
                body, request.headers.get("x-session-id", "").strip()
            )
            models = service.chains.resolve(parsed.model, client.default_model)
        except (RequestError, ConfigError) as exc:
            return _error(400, str(exc), "invalid_request_error")
        hint = ConversationHint(client.name, parsed.session)
        started = time.monotonic()

        def run(stream: bool):
            return service.chains.run(
                models,
                parsed.messages,
                stream=stream,
                conversation=hint,
                params=parsed.params,
            )

        try:
            result = await run_in_threadpool(run, parsed.stream)
        except GatewayError as exc:
            return gateway_error_response(exc, service)
        except ConfigError as exc:
            return _error(400, str(exc), "invalid_request_error")

        if isinstance(result, ChatResponse):
            _log_turn(client.name, models, result, started)
            return JSONResponse(
                completion_json(result, parsed.messages), headers=_headers(result)
            )
        return StreamingResponse(
            _event_stream(result, parsed, client.name, models, started),
            media_type="text/event-stream",
            headers={
                **_headers(result),
                "Cache-Control": "no-cache",
                "X-Accel-Buffering": "no",
            },
        )

    async def _event_stream(stream: StreamBase, parsed, client_name, models, started):
        completion_id = new_id()
        model = stream.origin.model
        iterator = iter(stream)
        done = object()

        def pull():
            try:
                return next(iterator)
            except StopIteration:
                return done

        pending: asyncio.Future | None = None
        try:
            yield sse(
                chunk_json(completion_id, model, {"role": "assistant", "content": ""})
            )
            while True:
                pending = asyncio.ensure_future(run_in_threadpool(pull))
                while True:
                    finished, _ = await asyncio.wait({pending}, timeout=KEEPALIVE)
                    if finished:
                        break
                    yield ": keep-alive\n\n"
                chunk = pending.result()
                pending = None
                if chunk is done:
                    break
                if chunk.delta:
                    yield sse(
                        chunk_json(completion_id, model, {"content": chunk.delta})
                    )
            response = stream.collect()
            reason = response.finish_reason or "stop"
            if response.tool_calls:
                calls = tool_calls_json(response.tool_calls, indexed=True)
                yield sse(chunk_json(completion_id, model, {"tool_calls": calls}))
                reason = "tool_calls"
            usage = (
                estimate_usage(parsed.messages, response)
                if parsed.include_usage
                else None
            )
            yield sse(chunk_json(completion_id, model, {}, reason, usage))
            yield "data: [DONE]\n\n"
            _log_turn(client_name, models, response, started)
        except StreamBroken as exc:
            log.warning("поток оборван: %s", exc)
            yield sse(error_json(str(exc), "stream_error"))
            yield "data: [DONE]\n\n"
        finally:
            # Клиент ушёл или поток кончился. Если поток ещё читается в потоке
            # пула — закрыть его можно только когда тот вернётся.
            if pending is not None and not pending.done():
                pending.add_done_callback(lambda _f: stream.close())
            else:
                stream.close()

    async def persist_loop() -> None:
        store = service.store
        state = service.runner.gateway.state
        while True:
            await asyncio.sleep(PERSIST_EVERY)
            if store is not None:
                await run_in_threadpool(persist.dump, state, store)

    @asynccontextmanager
    async def lifespan(app: Starlette) -> AsyncIterator[None]:
        store = service.store
        state = service.runner.gateway.state
        if store is not None:
            restored = persist.load(state, store)
            if restored:
                log.info("поднято сохранённых пауз: %d", restored)
        task = asyncio.create_task(persist_loop())
        try:
            yield
        finally:
            task.cancel()
            if store is not None:
                persist.dump(state, store)
            service.close()

    routes = [
        Route("/health", health),
        Route("/v1/models", list_models),
        Route("/v1/chat/completions", chat, methods=["POST"]),
        Route("/status", get_status),
        Route("/admin/reload", reload, methods=["POST"]),
    ]
    return Starlette(routes=routes, lifespan=lifespan)
