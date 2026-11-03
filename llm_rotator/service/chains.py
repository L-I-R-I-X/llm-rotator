"""Обход цепочки моделей на один ход.

То, что README ядра называет «задачей вызывающего кода»: на каждую модель
цепочки — `ModelRunner.send`, ошибка модели — следующая модель, а лимит
времени и попыток общий на весь ход (один `TurnBudget`).
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import replace

from ..gateway import Attempt, Failure, GatewayError
from ..registry import ConfigError
from ..runner import ModelRunner
from ..transport import ChatResponse, ConversationHint, StreamBase
from .clients import CHAIN_PREFIX


class ChainRunner:
    def __init__(self, runner: ModelRunner) -> None:
        self.runner = runner

    def resolve(self, requested: str, default: str) -> tuple[str, ...]:
        """Имена моделей по порядку. Не указано или неизвестно — `default`."""
        for ref in (requested, default):
            models = self._try(ref)
            if models:
                return models
        raise ConfigError(
            "модель не указана, а у клиента нет default_model "
            if not requested and not default
            else f"модель {requested!r} неизвестна, а default_model не задан"
        )

    def _try(self, ref: str) -> tuple[str, ...]:
        registry = self.runner.registry
        if not ref:
            return ()
        if ref.startswith(CHAIN_PREFIX):
            chain = registry.chains.get(ref.removeprefix(CHAIN_PREFIX))
            return chain.model_names if chain else ()
        model = registry.models.get(ref)
        return (model.name,) if model and model.enabled else ()

    def run(
        self,
        models: Sequence[str],
        messages: Sequence[dict],
        *,
        stream: bool,
        conversation: ConversationHint | None,
        params: dict[str, object],
        read_timeout: float | None = None,
    ) -> ChatResponse | StreamBase:
        runner = self.runner
        gateway = runner.gateway
        budget = gateway.turn_budget()
        factory = runner.executor_for(stream)
        attempts: list[Attempt] = []
        last: GatewayError | None = None
        for name in models:
            request = runner.request(name, messages, read_timeout, params, conversation)
            try:
                result = runner.send(
                    runner.registry.routes(name), name, request, factory, budget
                )
            except GatewayError as exc:
                attempts.extend(exc.attempts)
                last = exc
                if any(a.failure is Failure.FATAL for a in exc.attempts):
                    break  # запрос сам плох: другая модель его не исправит
                if budget.exhausted():
                    break
            else:
                return _with_earlier(result, attempts)
        raise gateway.chain_error(attempts, models, last)


def _with_earlier(
    result: ChatResponse | StreamBase, earlier: Sequence[Attempt]
) -> ChatResponse | StreamBase:
    """Отказы прежних моделей цепочки — тоже часть истории этого ответа."""
    if not earlier:
        return result
    if isinstance(result, ChatResponse):
        return replace(result, attempts=(*earlier, *result.attempts))
    result.attempts = (*earlier, *result.attempts)
    return result
