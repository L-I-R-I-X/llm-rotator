"""Снимок состояния для `/status`: паузы, занятость, здоровье, очередь удаления.

Здесь нет путей к файлам сессий и значений ключей — только имена связок.
"""

from __future__ import annotations

from ..gateway import key_target, proxy_target, route_target
from ..runner import ModelRunner


def snapshot(runner: ModelRunner) -> dict:
    registry = runner.registry
    state = runner.gateway.state
    context = runner._context
    bindings = []
    for provider_name, provider in registry.providers.items():
        for binding in registry.bindings_for(provider_name):
            proxy = binding.proxy.name
            pauses = {
                "key": state.left(key_target(binding.key_name)),
                "proxy": state.left(proxy_target(proxy)),
                "route": state.left(route_target(proxy, provider_name)),
            }
            cooldown = state.cooldown_keys.get(binding.key_name)
            if cooldown is not None:
                pauses["quota"] = max(0.0, cooldown - state.clock())
            active = {k: round(v, 1) for k, v in pauses.items() if v > 0}
            entry = {
                "binding": str(binding),
                "provider": provider_name,
                "paused": active,
            }
            if provider.web is not None:
                entry["max_concurrent"] = binding.max_concurrent
                if context is not None:
                    entry["busy"] = context.busy.count(binding.key_name)
                    entry["account"] = context.store.account_health(binding.key_name)
            bindings.append(entry)
    suspended = {
        name: round(until - state.clock(), 1)
        for name, until in state.suspended.items()
        if until > state.clock()
    }
    out: dict = {"bindings": bindings, "suspended_providers": suspended}
    if context is not None:
        out["chats"] = {
            s: len(context.store.by_state(s))
            for s in ("active", "busy", "abandoned", "deleted", "delete_failed")
        }
    out["warnings"] = list(registry.warnings)
    return out
