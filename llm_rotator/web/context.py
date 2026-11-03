"""Общее состояние веб-транспортов: хранилище, пул клиентов, занятость связок."""

from __future__ import annotations

import threading
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path

import httpx

from ..registry import Proxy
from .busy import BusyTracker
from .store import ConversationStore

HttpFactory = Callable[[Proxy], httpx.Client]

STATE_DIR = "state"
DB_NAME = "rotator.db"
CONNECT_TIMEOUT = 10.0


def default_http(proxy: Proxy) -> httpx.Client:
    """Клиент через прокси связки. Таймауты чтения задаёт сам клиент сайта."""
    return httpx.Client(
        proxy=proxy.httpx_proxy,
        timeout=httpx.Timeout(30.0, connect=CONNECT_TIMEOUT),
    )


@dataclass
class TransportContext:
    store: ConversationStore
    busy: BusyTracker = field(default_factory=BusyTracker)
    http_factory: HttpFactory = default_http
    pow_solver: Callable | None = None
    """Подмена решателя PoW (тесты); `None` — настоящий, через wasmtime."""
    _clients: dict[str, httpx.Client] = field(default_factory=dict, repr=False)
    _lock: threading.Lock = field(default_factory=threading.Lock, repr=False)

    @classmethod
    def default(cls, base_dir: Path | None = None) -> TransportContext:
        base = base_dir if base_dir is not None else Path.cwd()
        return cls(ConversationStore(base / STATE_DIR / DB_NAME))

    def http(self, proxy: Proxy) -> httpx.Client:
        """Один клиент на имя прокси: в нём живёт пул соединений."""
        with self._lock:
            client = self._clients.get(proxy.name)
            if client is None:
                client = self._clients[proxy.name] = self.http_factory(proxy)
            return client

    def close(self) -> None:
        with self._lock:
            clients = list(self._clients.values())
            self._clients.clear()
        for client in clients:
            client.close()
        self.store.close()
