"""`llm-rotator doctor`: проверка связок. Только чтение, ничего не тратит.

Для каждой связки: прокси отвечает; у веб-аккаунта жив токен (`ping`); у
API-ключа принимается запрос списка моделей. Мёртвые веб-аккаунты помечаются
`needs_auth` в хранилище, и для них подсказывается команда перелогина.
"""

from __future__ import annotations

import socket
from dataclasses import dataclass
from urllib.parse import urlparse

import httpx

from .gateway import adapter_for
from .registry import Binding, ConfigError, Provider, Registry

TCP_TIMEOUT = 5.0
HTTP_TIMEOUT = 15.0


@dataclass(frozen=True)
class Row:
    binding: str
    provider: str
    ok: bool
    detail: str
    hint: str = ""


def proxy_reachable(binding: Binding) -> str:
    """Пустая строка — отвечает (или прямое соединение); иначе причина."""
    address = binding.proxy.httpx_proxy
    if not address:
        return ""
    parsed = urlparse(address)
    try:
        with socket.create_connection((parsed.hostname, parsed.port or 0), TCP_TIMEOUT):
            return ""
    except OSError as exc:
        return f"прокси {binding.proxy.name} не отвечает ({type(exc).__name__})"


def check_web(provider: Provider, binding: Binding, http: httpx.Client, store) -> Row:
    from .web.deepseek.account import load_account
    from .web.deepseek.client import DeepSeekClient
    from .web.errors import WebError

    assert provider.web is not None
    name = str(binding)
    try:
        client = DeepSeekClient(
            load_account(binding.key_value), http, provider.web, provider.base_url
        )
        client.ping()
    except WebError as exc:
        if exc.kind == "auth":
            if store is not None:
                store.mark_account(binding.key_name, "needs_auth", exc.detail)
            return Row(
                name,
                provider.name,
                False,
                "токен не принят (needs_auth)",
                f"llm-rotator auth deepseek {binding.key_name}",
            )
        return Row(name, provider.name, False, f"{exc.kind}: {exc.detail}")
    if store is not None:
        store.mark_account(binding.key_name, "ok")
    return Row(name, provider.name, True, "токен жив")


def check_api(provider: Provider, binding: Binding, http: httpx.Client) -> Row:
    name = str(binding)
    if not provider.models_url:
        return Row(name, provider.name, True, "проверять нечем (нет models_url)")
    headers = dict(provider.headers)
    if provider.requires_key:
        headers.update(adapter_for(provider).headers(binding.key_value))
    try:
        response = http.get(provider.models_url, headers=headers, timeout=HTTP_TIMEOUT)
    except httpx.HTTPError as exc:
        return Row(name, provider.name, False, f"{type(exc).__name__}")
    if response.status_code == 200:
        return Row(name, provider.name, True, "список моделей получен")
    if response.status_code in (401, 403):
        return Row(
            name, provider.name, False, f"ключ не принят (HTTP {response.status_code})"
        )
    return Row(name, provider.name, False, f"HTTP {response.status_code}")


def run(registry: Registry, store=None, http_for=None) -> list[Row]:
    """Проверить все связки. `http_for(proxy)` подменяет клиентов (тесты)."""
    own: dict[str, httpx.Client] = {}

    def client(proxy) -> httpx.Client:
        if http_for is not None:
            return http_for(proxy)
        if proxy.name not in own:
            own[proxy.name] = httpx.Client(
                proxy=proxy.httpx_proxy, timeout=HTTP_TIMEOUT
            )
        return own[proxy.name]

    rows: list[Row] = []
    try:
        for provider_name, provider in registry.providers.items():
            for binding in registry.bindings_for(provider_name):
                problem = proxy_reachable(binding)
                if problem:
                    rows.append(Row(str(binding), provider_name, False, problem))
                    continue
                http = client(binding.proxy)
                try:
                    if provider.web is not None:
                        rows.append(check_web(provider, binding, http, store))
                    else:
                        rows.append(check_api(provider, binding, http))
                except ConfigError as exc:
                    rows.append(Row(str(binding), provider_name, False, str(exc)))
    finally:
        for http in own.values():
            http.close()
    return rows


def direct_web_accounts(registry: Registry) -> list[str]:
    """Веб-аккаунты без прокси: вход и работа идут с прямого IP."""
    out = []
    for name, provider in registry.providers.items():
        if provider.web is None:
            continue
        out += [str(b) for b in registry.bindings_for(name) if b.proxy.is_direct]
    return out


def render(rows: list[Row], direct: list[str]) -> str:
    width = max((len(r.binding) for r in rows), default=10)
    lines = []
    for row in rows:
        mark = "OK  " if row.ok else "FAIL"
        lines.append(f"{mark} {row.binding:<{width}}  {row.provider}: {row.detail}")
        if row.hint:
            lines.append(f"     → {row.hint}")
    if direct:
        lines.append("")
        lines.append(
            "Веб-аккаунты на прямом IP (прокси не закреплён): " + ", ".join(direct)
        )
    return "\n".join(lines)
