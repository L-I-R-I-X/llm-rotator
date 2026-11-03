"""Уборка удалённых чатов (R1): ротатор удаляет за собой то, что создал сам.

Берёт только записи из таблицы `chats`: брошенные и давно простаивающие. Чатов,
которых там нет, не касается — чужое пользователь чистит сам. Удаляет через
ту же связку (аккаунт и прокси), с которой чат создан.
"""

from __future__ import annotations

import logging
import os
import threading
import time
from collections.abc import Callable

from ..registry import Binding, Provider, Registry
from .context import TransportContext
from .errors import WebError
from .store import ChatRecord

log = logging.getLogger(__name__)

MIN_GAP = 2.0
"""Между удалениями на один аккаунт — не чаще, чтобы не ловить лимиты."""
RETRY_PAUSES = (60.0, 300.0, 1800.0)
"""Паузы между повторами; дальше — последняя. Пятая неудача — `delete_failed`."""
MAX_ATTEMPTS = 5
DELETED_KEEP = 7 * 24 * 3600.0
BUSY_INTERVAL = 2.0
IDLE_INTERVAL = 30.0


def _deepseek(provider: Provider, binding: Binding, context: TransportContext):
    from .deepseek.account import load_account
    from .deepseek.client import DeepSeekClient

    assert provider.web is not None
    return DeepSeekClient(
        load_account(binding.key_value),
        context.http(binding.proxy),
        provider.web,
        provider.base_url,
    )


DELETERS: dict[str, Callable] = {"deepseek_web": _deepseek}
"""api_style → фабрика клиента с методом `delete_chat(chat_id)`."""


class Janitor:
    def __init__(
        self,
        registry: Registry,
        context: TransportContext,
        clock: Callable[[], float] = time.time,
    ) -> None:
        self._registry = registry
        self._context = context
        self._clock = clock
        self._last_delete: dict[str, float] = {}
        self._wake = threading.Event()
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None

    def set_registry(self, registry: Registry) -> None:
        self._registry = registry

    # --- поток --------------------------------------------------------------

    def start(self) -> None:
        if self._thread is not None:
            return
        self._thread = threading.Thread(
            target=self._loop, name="llm-rotator-janitor", daemon=True
        )
        self._thread.start()

    def wake(self) -> None:
        """Есть что убирать — не ждать конца паузы."""
        self._wake.set()

    def stop(self) -> None:
        self._stop.set()
        self._wake.set()
        if self._thread is not None:
            self._thread.join(timeout=5)
            self._thread = None

    def _loop(self) -> None:
        while not self._stop.is_set():
            try:
                busy = self.run_once() > 0
            except Exception:
                log.exception("janitor: сбой прохода")
                busy = False
            self._wake.wait(BUSY_INTERVAL if busy else IDLE_INTERVAL)
            self._wake.clear()

    # --- проход -------------------------------------------------------------

    def _idle_ttl(self) -> dict[str, float]:
        return {
            name: provider.web.chat_idle_ttl
            for name, provider in self._registry.providers.items()
            if provider.web is not None and provider.web.delete_policy != "never"
        }

    def run_once(self) -> int:
        """Один проход. Возвращает число удалённых чатов."""
        store = self._context.store
        now = self._clock()
        store.purge_deleted(now - DELETED_KEEP)
        deleted = 0
        touched: set[str] = set()
        for record in store.due_for_deletion(now, self._idle_ttl()):
            if record.key_name in touched:
                continue  # одно удаление на аккаунт за проход
            if self._delete(record, now):
                touched.add(record.key_name)
                deleted += 1
        return deleted

    def _binding(self, record: ChatRecord) -> Binding | None:
        """Связка чата. Если прокси ключа с тех пор сменили — любая его текущая."""
        bindings = [
            b
            for b in self._registry.bindings_for(record.provider)
            if b.key_name == record.key_name
        ]
        for binding in bindings:
            if binding.proxy.name == record.proxy_name:
                return binding
        return bindings[0] if bindings else None

    def _delete(self, record: ChatRecord, now: float) -> bool:
        store = self._context.store
        provider = self._registry.providers.get(record.provider)
        binding = self._binding(record)
        if provider is None or binding is None or provider.api_style not in DELETERS:
            # Ключ убрали из конфигурации: удалить нечем, и не станет.
            log.warning(
                "janitor: чат %s (%s) не удалить — нет связки %s",
                record.chat_id,
                record.provider,
                record.key_name,
            )
            store.record_delete_failure(record.provider, record.chat_id, now, True)
            return False
        if not self._account_ready(record.key_name, binding):
            return False
        if now - self._last_delete.get(record.key_name, -MIN_GAP) < MIN_GAP:
            return False
        self._last_delete[record.key_name] = now
        try:
            client = DELETERS[provider.api_style](provider, binding, self._context)
            client.delete_chat(record.chat_id)  # False — уже нет: тоже конец
        except WebError as exc:
            self._failed(record, exc, now)
            return False
        store.mark_deleted(record.provider, record.chat_id)
        if store.account_health(record.key_name) != "ok":
            store.mark_account(record.key_name, "ok")
        return True

    def _account_ready(self, key_name: str, binding: Binding) -> bool:
        """Аккаунт с мёртвым токеном не трогаем, пока файл сессии не обновят."""
        health, marked_at = self._context.store.account_state(key_name)
        if health == "ok":
            return True
        try:
            return os.stat(binding.key_value).st_mtime > marked_at
        except OSError:
            return False

    def _failed(self, record: ChatRecord, error: WebError, now: float) -> None:
        store = self._context.store
        if error.kind == "auth":
            store.mark_account(record.key_name, "needs_auth", error.detail)
            return  # ждём перелогина, попытку не считаем
        attempts = record.delete_attempts + 1
        pause = RETRY_PAUSES[min(attempts - 1, len(RETRY_PAUSES) - 1)]
        log.info(
            "janitor: удалить %s не вышло (%s), попытка %d",
            record.chat_id,
            error.kind,
            attempts,
        )
        store.record_delete_failure(
            record.provider, record.chat_id, now + pause, attempts >= MAX_ATTEMPTS
        )
