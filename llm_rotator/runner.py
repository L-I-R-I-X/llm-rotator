"""Вызов одной модели: обход её провайдеров, ключей и адресов.

`ModelRunner` — нижняя половина клиента: умеет вызвать ровно одну модель
(`complete`) или открыть её поток (`stream`), перебирая связки (ключ, прокси) до
первого ответа. Выбор между моделями — цепочки ролей, привязка диалога,
подтверждения — сюда не входит: это надстройка вызывающего кода, которая зовёт
`send()` на каждую модель цепочки.

Экземпляр держит реестр, шлюз и транспорты и переживает всю сессию: следующий
запрос не станет снова биться в уже признанный мёртвым ключ.

Разделение по слоям сделано так, чтобы расширение не трогало этот файл: новый
OpenAI-совместимый провайдер — это строчка в providers.toml, несовместимый API —
класс в transport.TRANSPORTS, а вызывающий код продолжает видеть те же
ChatResponse и Origin.
"""

from __future__ import annotations

import logging
from collections.abc import Callable, Sequence
from dataclasses import replace
from typing import TYPE_CHECKING, Self

from .gateway import (
    Binding,
    Executor,
    Gateway,
    Route,
    SessionState,
    TurnBudget,
)
from .registry import Provider, Registry, load_registry
from .transport import (
    ChatRequest,
    ChatResponse,
    ConversationHint,
    StreamBase,
    Transport,
    make_transport,
)

if TYPE_CHECKING:
    from .web.context import TransportContext
    from .web.janitor import Janitor

log = logging.getLogger(__name__)


class ModelRunner:
    """Живой доступ к моделям: обход провайдеров плюс разбор ответа.

    Один экземпляр держит и шлюз, и транспорты, и состояние сессии (какие
    ключи и адреса уже признаны нерабочими), поэтому переживает всю сессию:
    следующий запрос не станет снова биться в тот же мёртвый ключ.
    """

    def __init__(
        self,
        registry: Registry | None = None,
        gateway: Gateway | None = None,
        state: SessionState | None = None,
        transports: dict[str, Transport] | None = None,
        context: TransportContext | None = None,
        janitor: bool = True,
    ) -> None:
        """
        `registry` и `state` — необязательные: без них берутся из конфигурации
        базового каталога. `state` имеет смысл только когда шлюз не передан: у
        переданного шлюза своё состояние, и подмена ему ничего не даст.
        `transports` подменяет транспорты по api_style — это точка расширения
        для тестов и несовместимых API. `context` — общее состояние веб-чатов
        (хранилище, пул клиентов); без него создаётся при первом обращении к
        веб-провайдеру, в `state/` рядом с `models/`. `janitor=False` не
        запускает фоновое удаление брошенных веб-чатов (отладка, тесты).
        """
        self.registry = registry if registry is not None else load_registry()
        self.gateway = (
            gateway if gateway is not None else Gateway(self.registry, state=state)
        )
        self._transports: dict[str, Transport] = dict(transports or {})
        self._context = context
        self._janitor_enabled = janitor
        self._janitor: Janitor | None = None
        self._closed = False

    def transport_for(self, provider: Provider) -> Transport:
        """Транспорт для провайдера, один на api_style.

        Кэш нужен из-за пула соединений внутри openai-клиента: транспорт, а
        значит и клиент на каждую связку, поднимается один раз. Неизвестный
        стиль — ConfigError прямо здесь, до отправки.
        """
        transport = self._transports.get(provider.api_style)
        if transport is None:
            context = self._web_context() if provider.web is not None else None
            transport = make_transport(provider, self.gateway, context)
            self._transports[provider.api_style] = transport
        return transport

    def _web_context(self) -> TransportContext:
        if self._context is None:
            from .web.context import TransportContext

            self._context = TransportContext.default(self.registry.base_dir)
        if self._janitor is None and self._janitor_enabled:
            from .web.janitor import Janitor

            self._janitor = Janitor(self.registry, self._context)
            self._context.wake = self._janitor.wake
            self._janitor.start()
        return self._context

    def complete(
        self,
        model_name: str,
        messages: Sequence[dict],
        *,
        read_timeout: float | None = None,
        conversation: ConversationHint | None = None,
        **params: object,
    ) -> ChatResponse:
        """Вызвать одну модель, минуя цепочки ролей. Бросает GatewayError."""
        model = self.registry.model(model_name)
        response = self._send(
            self.registry.routes(model.name),
            model.name,
            self._request(model.name, messages, read_timeout, params, conversation),
            self._executor,
        )
        assert isinstance(response, ChatResponse)
        return response

    def stream(
        self,
        model_name: str,
        messages: Sequence[dict],
        *,
        read_timeout: float | None = None,
        conversation: ConversationHint | None = None,
        **params: object,
    ) -> StreamBase:
        """Открыть поток ответа одной модели.

        Обход идёт до первого чанка: пока наружу не отдано ни одного символа,
        можно уйти на другой ключ или адрес. Дальше — нет, обрыв в середине
        стрима это StreamBroken.
        """
        model = self.registry.model(model_name)
        response = self._send(
            self.registry.routes(model.name),
            model.name,
            self._request(model.name, messages, read_timeout, params, conversation),
            self._executor_stream,
        )
        assert isinstance(response, StreamBase)
        return response

    def send(
        self,
        routes: Sequence[Route],
        model_name: str,
        request: ChatRequest,
        executor_factory: Callable[[ChatRequest], Executor],
        budget: TurnBudget | None = None,
    ) -> ChatResponse | StreamBase:
        """Обойти маршруты одной модели. Точка входа для надстройки цепочек."""
        return self._send(routes, model_name, request, executor_factory, budget)

    def request(
        self,
        model_name: str,
        messages: Sequence[dict],
        read_timeout: float | None,
        params: dict[str, object],
        conversation: ConversationHint | None = None,
    ) -> ChatRequest:
        """Собрать тело запроса к модели: сообщения плюс потолок из конфига."""
        return self._request(model_name, messages, read_timeout, params, conversation)

    def _request(
        self,
        model_name: str,
        messages: Sequence[dict],
        read_timeout: float | None,
        params: dict[str, object],
        conversation: ConversationHint | None = None,
    ) -> ChatRequest:
        """Тело запроса к модели: сообщения плюс параметры из конфигурации.

        max_tokens берётся из models.toml (Model.max_output_tokens) и нужен для
        прохода проверки провайдера, а не для краткости: Groq сверяет заявленный
        потолок с лимитом OTPM и отклоняет запрос целиком, если потолок выше
        лимита. Явный max_tokens от вызывающего перекрывает значение из
        конфига — иначе нельзя было бы попросить длинный ответ.
        """
        cap = self.registry.model(model_name).max_output_tokens
        if cap is not None and "max_tokens" not in params:
            params = {**params, "max_tokens": cap}
        return ChatRequest(
            model=model_name,
            messages=list(messages),
            params=params,
            read_timeout=read_timeout,
            conversation=conversation,
        )

    def _send(
        self,
        routes: Sequence[Route],
        model_name: str,
        request: ChatRequest,
        executor_factory: Callable[[ChatRequest], Executor],
        budget: TurnBudget | None = None,
    ) -> ChatResponse | StreamBase:
        routes = self._order_web_bindings(routes, request)
        rotation = self.gateway.rotate(routes, executor_factory(request), budget)
        response = rotation.payload
        if not isinstance(response, (ChatResponse, StreamBase)):
            raise TypeError(
                f"{model_name}: транспорт вернул {type(response).__name__} вместо ответа"
            )
        if isinstance(response, ChatResponse):
            return replace(response, attempts=rotation.failed)
        response.attempts = rotation.failed
        return response

    def _order_web_bindings(
        self, routes: Sequence[Route], request: ChatRequest
    ) -> Sequence[Route]:
        """Порядок связок веб-провайдеров: где уже идёт этот разговор, затем
        самые давно простаивавшие свободные, затем занятые.

        Без этого весь трафик шёл бы в первый аккаунт из keys.toml. `rotate`
        сам переставляет «сомнительные» связки назад стабильной сортировкой,
        так что наш порядок среди надёжных сохраняется.
        """
        if self._context is None or not any(r.provider.web for r in routes):
            return routes
        from .web.fingerprint import fingerprint
        from .web.planner import find_continuation

        busy = self._context.busy
        store = self._context.store
        hint = request.conversation
        namespace = hint.namespace if hint else "default"
        session = hint.session if hint else ""
        fp = fingerprint(request.messages, request.params.get("tools"))
        ordered: list[Route] = []
        for route in routes:
            web = route.provider.web
            if web is None or len(route.bindings) < 2:
                ordered.append(route)
                continue

            def rank(binding: Binding, route: Route = route, web=web) -> tuple:
                free = busy.is_free(binding.key_name, binding.max_concurrent)
                affine = free and (
                    find_continuation(
                        store,
                        web,
                        provider=route.provider.name,
                        key_name=binding.key_name,
                        namespace=namespace,
                        session=session,
                        fp=fp,
                    )
                    is not None
                )
                return (
                    0 if affine else 1 if free else 2,
                    busy.last_used(binding.key_name),
                )

            ordered.append(
                replace(route, bindings=tuple(sorted(route.bindings, key=rank)))
            )
        return ordered

    def executor_for(self, stream: bool) -> Callable[[ChatRequest], Executor]:
        """Фабрика исполнителя попыток для `send()`: обычный вызов или поток."""
        return self._executor_stream if stream else self._executor

    def reload(self, registry: Registry) -> None:
        """Подменить реестр на лету (после правки keys.toml и т.п.).

        Паузы и аффинити остаются: они привязаны к именам, а не к объектам.
        """
        self.registry = registry
        self.gateway.registry = registry
        if self._janitor is not None:
            self._janitor.set_registry(registry)

    def _executor(self, request: ChatRequest) -> Executor:
        def attempt(route: Route, binding: Binding):
            return self.transport_for(route.provider).complete(
                route, binding, self._for_route(request, route)
            )

        return attempt

    def _executor_stream(self, request: ChatRequest) -> Executor:
        def attempt(route: Route, binding: Binding):
            return self.transport_for(route.provider).open_stream(
                route, binding, self._for_route(request, route)
            )

        return attempt

    def _for_route(self, request: ChatRequest, route: Route) -> ChatRequest:
        """Имя модели в теле — так, как его ждёт конкретный провайдер.

        Google в списке отдаёт models/gemini-3.8-flash, а в чат просит
        gemini-3.8-flash; при нескольких провайдерах у модели префиксы могут
        различаться, поэтому подставляется per-route.
        """
        name = route.provider.strip_id_prefix(route.model.name)
        if name == request.model:
            return request
        return replace(request, model=name)

    def forget_dialog(self) -> None:
        """Забыть кэш «у провайдера нет модели».

        Стираются пометки «у провайдера нет модели», но не баны ключей и
        адресов: если адрес не отвечал, в новом диалоге он тоже не ответит.
        """
        self.gateway.state.forget_models()

    def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        for transport in self._transports.values():
            transport.close()
        self._transports.clear()
        if self._janitor is not None:
            self._janitor.stop()
        if self._context is not None:
            self._context.close()
        self.gateway.close()

    def __enter__(self) -> Self:
        return self

    def __exit__(self, *exc_info: object) -> None:
        self.close()

    def __str__(self) -> str:
        return (
            f"ModelRunner: провайдеров {len(self.registry.providers)}, "
            f"моделей {len(self.registry.models)}, "
            f"транспортов {len(self._transports)}"
        )


__all__ = ["ModelRunner"]
