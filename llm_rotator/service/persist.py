"""Паузы `SessionState` переживают рестарт: не бить снова в мёртвые аккаунты.

Моментальные часы (`monotonic`) между запусками несравнимы, поэтому на диск
идёт оставшееся время; при чтении оно прибавляется к текущему моменту.
"""

from __future__ import annotations

import json
import time

from ..gateway import Penalty, SessionState
from ..web.store import ConversationStore

MIN_LEFT = 1.0


def dump(state: SessionState, store: ConversationStore) -> int:
    """Записать действующие паузы. Возвращает их число."""
    now = state.clock()
    rows: list[tuple[str, str, float, int]] = []
    for target, penalty in list(state.penalties.items()):
        if penalty.until - now > MIN_LEFT:
            rows.append(
                (
                    "penalty",
                    json.dumps(list(target)),
                    penalty.until - now,
                    penalty.failures,
                )
            )
    for name, until in list(state.cooldown_keys.items()):
        if until - now > MIN_LEFT:
            rows.append(("cooldown", name, until - now, 0))
    for name, until in list(state.suspended.items()):
        if until - now > MIN_LEFT:
            rows.append(("suspended", name, until - now, 0))
    store.replace_pauses(rows, time.time())
    return len(rows)


def load(state: SessionState, store: ConversationStore) -> int:
    """Поднять сохранённые паузы с поправкой на время простоя."""
    saved_at, rows = store.pauses()
    if not rows:
        return 0
    elapsed = max(0.0, time.time() - saved_at)
    now = state.clock()
    restored = 0
    for kind, target, left, failures in rows:
        left -= elapsed
        if left <= MIN_LEFT:
            continue
        if kind == "penalty":
            state.penalties[tuple(json.loads(target))] = Penalty(failures, now + left)
        elif kind == "cooldown":
            state.cooldown_keys[target] = now + left
        elif kind == "suspended":
            state.suspended[target] = now + left
        restored += 1
    return restored
