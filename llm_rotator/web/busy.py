"""Занятость связок: не больше `max_concurrent` одновременных ходов на аккаунт."""

from __future__ import annotations

import threading


class BusyTracker:
    def __init__(self) -> None:
        self._counts: dict[str, int] = {}
        self._lock = threading.Lock()

    def acquire(self, key_name: str, limit: int | None) -> bool:
        """Занять слот. `limit=None` — без ограничения. False — слотов нет."""
        with self._lock:
            used = self._counts.get(key_name, 0)
            if limit is not None and used >= limit:
                return False
            self._counts[key_name] = used + 1
            return True

    def release(self, key_name: str) -> None:
        with self._lock:
            left = self._counts.get(key_name, 0) - 1
            if left > 0:
                self._counts[key_name] = left
            else:
                self._counts.pop(key_name, None)

    def count(self, key_name: str) -> int:
        with self._lock:
            return self._counts.get(key_name, 0)
