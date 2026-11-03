"""Хранилище веб-слоя (SQLite): удалённые чаты, кэш файлов, здоровье аккаунтов.

Один файл, общее соединение под замком, режим WAL: ходы идут из пула потоков.
Здесь живёт только то, что ротатор создал сам, — других чатов реестр не знает
и трогать их нельзя (чистка чужих чатов — дело пользователя).
"""

from __future__ import annotations

import json
import sqlite3
import threading
import time
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from pathlib import Path

ACTIVE = "active"
BUSY = "busy"
ABANDONED = "abandoned"
DELETED = "deleted"
DELETE_FAILED = "delete_failed"
STATES = (ACTIVE, BUSY, ABANDONED, DELETED, DELETE_FAILED)

_SCHEMA = """
CREATE TABLE IF NOT EXISTS chats (
    provider TEXT NOT NULL,
    chat_id TEXT NOT NULL,
    key_name TEXT NOT NULL,
    proxy_name TEXT NOT NULL,
    namespace TEXT NOT NULL,
    session TEXT NOT NULL,
    context_key TEXT NOT NULL,
    sent_keys TEXT NOT NULL,
    parent_message_id TEXT,
    turns INTEGER NOT NULL DEFAULT 0,
    remote_chars INTEGER NOT NULL DEFAULT 0,
    created_at REAL NOT NULL,
    last_used_at REAL NOT NULL,
    state TEXT NOT NULL,
    delete_attempts INTEGER NOT NULL DEFAULT 0,
    next_attempt_at REAL NOT NULL DEFAULT 0,
    PRIMARY KEY (provider, chat_id)
);
CREATE INDEX IF NOT EXISTS chats_lookup
    ON chats (provider, key_name, namespace, session, state);
CREATE TABLE IF NOT EXISTS files (
    key_name TEXT NOT NULL,
    sha256 TEXT NOT NULL,
    file_id TEXT NOT NULL,
    created_at REAL NOT NULL,
    PRIMARY KEY (key_name, sha256)
);
CREATE TABLE IF NOT EXISTS pauses (
    kind TEXT NOT NULL,
    target TEXT NOT NULL,
    left REAL NOT NULL,
    failures INTEGER NOT NULL DEFAULT 0,
    saved_at REAL NOT NULL
);
CREATE TABLE IF NOT EXISTS accounts (
    key_name TEXT PRIMARY KEY,
    health TEXT NOT NULL,
    detail TEXT NOT NULL DEFAULT '',
    updated_at REAL NOT NULL
);
"""


@dataclass(frozen=True, slots=True)
class ChatRecord:
    provider: str
    chat_id: str
    key_name: str
    proxy_name: str
    namespace: str
    session: str
    context_key: str
    sent_keys: tuple[str, ...]
    parent_message_id: int | str | None
    turns: int
    remote_chars: int
    created_at: float
    last_used_at: float
    state: str
    delete_attempts: int
    next_attempt_at: float = 0.0


_COLUMNS = (
    "provider, chat_id, key_name, proxy_name, namespace, session, context_key, "
    "sent_keys, parent_message_id, turns, remote_chars, created_at, last_used_at, "
    "state, delete_attempts, next_attempt_at"
)


def _record(row: Sequence) -> ChatRecord:
    return ChatRecord(
        provider=row[0],
        chat_id=row[1],
        key_name=row[2],
        proxy_name=row[3],
        namespace=row[4],
        session=row[5],
        context_key=row[6],
        sent_keys=tuple(json.loads(row[7])),
        parent_message_id=json.loads(row[8]) if row[8] is not None else None,
        turns=row[9],
        remote_chars=row[10],
        created_at=row[11],
        last_used_at=row[12],
        state=row[13],
        delete_attempts=row[14],
        next_attempt_at=row[15],
    )


class ConversationStore:
    def __init__(
        self, path: str | Path = ":memory:", clock: Callable[[], float] = time.time
    ) -> None:
        self._clock = clock
        if str(path) != ":memory:":
            Path(path).parent.mkdir(parents=True, exist_ok=True)
        self._db = sqlite3.connect(str(path), check_same_thread=False)
        self._lock = threading.RLock()
        with self._lock:
            if str(path) != ":memory:":
                self._db.execute("PRAGMA journal_mode=WAL")
            self._db.executescript(_SCHEMA)
            self._db.commit()

    def close(self) -> None:
        with self._lock:
            self._db.close()

    def _run(self, sql: str, args: Sequence = ()) -> sqlite3.Cursor:
        with self._lock:
            cursor = self._db.execute(sql, args)
            self._db.commit()
            return cursor

    def _all(self, sql: str, args: Sequence = ()) -> list[tuple]:
        with self._lock:
            return self._db.execute(sql, args).fetchall()

    # --- чаты ---------------------------------------------------------------

    def create_chat(
        self,
        provider: str,
        chat_id: str,
        key_name: str,
        proxy_name: str,
        namespace: str,
        session: str,
        context_key: str,
        state: str = BUSY,
    ) -> ChatRecord:
        now = self._clock()
        self._run(
            f"INSERT INTO chats ({_COLUMNS}) VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
            (
                provider,
                chat_id,
                key_name,
                proxy_name,
                namespace,
                session,
                context_key,
                "[]",
                None,
                0,
                0,
                now,
                now,
                state,
                0,
                0.0,
            ),
        )
        record = self.get(provider, chat_id)
        assert record is not None
        return record

    def get(self, provider: str, chat_id: str) -> ChatRecord | None:
        rows = self._all(
            f"SELECT {_COLUMNS} FROM chats WHERE provider=? AND chat_id=?",
            (provider, chat_id),
        )
        return _record(rows[0]) if rows else None

    def candidates(
        self, provider: str, key_name: str, namespace: str, session: str
    ) -> list[ChatRecord]:
        """Живые чаты этого аккаунта в этом пространстве имён и сессии."""
        rows = self._all(
            f"SELECT {_COLUMNS} FROM chats WHERE provider=? AND key_name=? "
            "AND namespace=? AND session=? AND state IN (?,?)",
            (provider, key_name, namespace, session, ACTIVE, BUSY),
        )
        return [_record(r) for r in rows]

    def claim(self, provider: str, chat_id: str) -> bool:
        """active → busy атомарно. False — чат уже занят другим ходом."""
        cursor = self._run(
            "UPDATE chats SET state=?, last_used_at=? "
            "WHERE provider=? AND chat_id=? AND state=?",
            (BUSY, self._clock(), provider, chat_id, ACTIVE),
        )
        return cursor.rowcount == 1

    def finish_turn(
        self,
        provider: str,
        chat_id: str,
        sent_keys: Sequence[str],
        parent_message_id: int | str | None,
        added_chars: int,
        state: str = ACTIVE,
    ) -> None:
        self._run(
            "UPDATE chats SET sent_keys=?, parent_message_id=?, turns=turns+1, "
            "remote_chars=remote_chars+?, last_used_at=?, state=? "
            "WHERE provider=? AND chat_id=?",
            (
                json.dumps(list(sent_keys)),
                json.dumps(parent_message_id),
                added_chars,
                self._clock(),
                state,
                provider,
                chat_id,
            ),
        )

    def set_state(self, provider: str, chat_id: str, state: str) -> None:
        assert state in STATES
        self._run(
            "UPDATE chats SET state=?, last_used_at=? WHERE provider=? AND chat_id=?",
            (state, self._clock(), provider, chat_id),
        )

    def abandon_conversation(
        self,
        provider: str,
        namespace: str,
        session: str,
        first_key: str,
        except_chat_id: str = "",
    ) -> int:
        """Брошенными считаем чаты того же разговора (по первому сообщению).

        Свободные (`active`) — занятые чужим ходом не трогаем. Вызывается после
        успешного хода: старые чаты этого разговора больше не продолжить —
        в запросе уже есть ответ, которого в них нет.
        """
        rows = self._all(
            "SELECT chat_id, sent_keys FROM chats WHERE provider=? AND namespace=? "
            "AND session=? AND state=? AND chat_id<>?",
            (provider, namespace, session, ACTIVE, except_chat_id),
        )
        stale = [
            chat_id
            for chat_id, sent in rows
            if (json.loads(sent) or [""])[0] == first_key
        ]
        for chat_id in stale:
            self.set_state(provider, chat_id, ABANDONED)
        return len(stale)

    def by_state(self, state: str, provider: str | None = None) -> list[ChatRecord]:
        sql = f"SELECT {_COLUMNS} FROM chats WHERE state=?"
        args: list = [state]
        if provider is not None:
            sql += " AND provider=?"
            args.append(provider)
        return [_record(r) for r in self._all(sql + " ORDER BY created_at", args)]

    # --- уборка -------------------------------------------------------------

    def due_for_deletion(
        self, now: float, idle_ttl: dict[str, float]
    ) -> list[ChatRecord]:
        """Чаты, которые пора удалить: брошенные и давно простаивающие.

        `idle_ttl` — провайдер → секунд простоя; провайдеров без записи (и с
        политикой `never`) здесь нет. Зависший в `busy` чат (процесс упал
        посреди хода) считается простаивающим так же, как `active`.
        """
        due: list[ChatRecord] = []
        for record in self.by_state(ABANDONED):
            if record.provider in idle_ttl and record.next_attempt_at <= now:
                due.append(record)
        for state in (ACTIVE, BUSY):
            for record in self.by_state(state):
                ttl = idle_ttl.get(record.provider)
                if ttl is not None and record.last_used_at < now - ttl:
                    due.append(record)
        return due

    def mark_deleted(self, provider: str, chat_id: str) -> None:
        self.set_state(provider, chat_id, DELETED)

    def record_delete_failure(
        self, provider: str, chat_id: str, next_attempt_at: float, final: bool
    ) -> None:
        """Неудачное удаление: попытка +1, повтор позже, после лимита — отказ.

        Чат при этом считается брошенным: простаивающий `active` больше не
        должен продолжаться, раз его пытаются удалить.
        """
        self._run(
            "UPDATE chats SET delete_attempts=delete_attempts+1, "
            "next_attempt_at=?, state=? WHERE provider=? AND chat_id=?",
            (next_attempt_at, DELETE_FAILED if final else ABANDONED, provider, chat_id),
        )

    def purge_deleted(self, older_than: float) -> int:
        cursor = self._run(
            "DELETE FROM chats WHERE state=? AND last_used_at<?", (DELETED, older_than)
        )
        return cursor.rowcount

    # --- файлы --------------------------------------------------------------

    def cached_file(self, key_name: str, sha256: str) -> str | None:
        rows = self._all(
            "SELECT file_id FROM files WHERE key_name=? AND sha256=?",
            (key_name, sha256),
        )
        return rows[0][0] if rows else None

    def remember_file(self, key_name: str, sha256: str, file_id: str) -> None:
        self._run(
            "INSERT OR REPLACE INTO files (key_name, sha256, file_id, created_at) "
            "VALUES (?,?,?,?)",
            (key_name, sha256, file_id, self._clock()),
        )

    # --- паузы сессии -------------------------------------------------------

    def replace_pauses(
        self, rows: Sequence[tuple[str, str, float, int]], saved_at: float
    ) -> None:
        with self._lock:
            self._db.execute("DELETE FROM pauses")
            self._db.executemany(
                "INSERT INTO pauses (kind, target, left, failures, saved_at) "
                "VALUES (?,?,?,?,?)",
                [(*row, saved_at) for row in rows],
            )
            self._db.commit()

    def pauses(self) -> tuple[float, list[tuple[str, str, float, int]]]:
        rows = self._all("SELECT kind, target, left, failures, saved_at FROM pauses")
        if not rows:
            return 0.0, []
        return rows[0][4], [(k, t, left, f) for k, t, left, f, _ in rows]

    # --- аккаунты -----------------------------------------------------------

    def mark_account(self, key_name: str, health: str, detail: str = "") -> None:
        self._run(
            "INSERT OR REPLACE INTO accounts (key_name, health, detail, updated_at) "
            "VALUES (?,?,?,?)",
            (key_name, health, detail, self._clock()),
        )

    def account_health(self, key_name: str) -> str:
        return self.account_state(key_name)[0]

    def account_state(self, key_name: str) -> tuple[str, float]:
        """(здоровье, когда отмечено). Нет записи — (`ok`, 0)."""
        rows = self._all(
            "SELECT health, updated_at FROM accounts WHERE key_name=?", (key_name,)
        )
        return (rows[0][0], rows[0][1]) if rows else ("ok", 0.0)
