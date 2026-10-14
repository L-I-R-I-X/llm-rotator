"""llm-rotator — ротация ключей, моделей и прокси для LLM-провайдеров.

Пакет ничего не знает про конкретный проект: он читает конфигурацию из каталога,
который ему указали, и умеет пройти по маршрутам модели, перебирая ключи и адреса
до первого ответа.

Устройство:

* ``registry``   — TOML/.env, валидация, ``Registry`` и её связки.
* ``gateway``    — правила отказа: что считать сломанным и куда идти дальше.
* ``transport``  — одна попытка на связке (OpenAI-совместимый слой).
* ``runner``     — вызов одной модели с обходом её провайдеров/ключей/адресов.
* ``discovery``  — список моделей у провайдеров (ручная утилита).

Пример:

    from llm_rotator import load_registry, ModelRunner

    registry = load_registry(base_dir="/path/to/project")
    with ModelRunner(registry) as runner:
        response = runner.complete("gpt-oss-120b", messages)
"""

from __future__ import annotations

from . import discovery, gateway, registry, runner, transport
from .discovery import (
    Discovered,
    cache_age,
    cache_is_fresh,
    find_extra_providers,
    load_discovered,
    merge_into_models,
    poll_all,
    poll_provider,
    print_report,
    read_cache,
    render_cache,
    run_startup_discovery,
    write_cache,
)
from .gateway import (
    API_STYLES,
    Attempt,
    Executor,
    Failure,
    Gateway,
    GatewayError,
    LLMError,
    Rotation,
    SessionState,
    TurnBudget,
    adapter_for,
    classify,
    error_detail,
    provider_error,
)
from .registry import (
    AUTH_STYLES,
    CHAINS_FILE,
    DEFAULT_API_STYLE,
    DEFAULT_AUTH,
    DEFAULT_CONTEXT_WINDOW,
    DEFAULT_PROXY_TYPE,
    DIRECT_PROXY,
    DISCOVERED_FILE,
    ENV_FILE,
    KEYS_FILE,
    MODELS_DIR,
    MODELS_FILE,
    PROVIDERS_FILE,
    PROXIES_FILE,
    ROLE_SUFFIX,
    ROOT,
    TURN_ATTEMPTS_VAR,
    TURN_DEADLINE_VAR,
    Binding,
    Chain,
    ConfigError,
    Key,
    Model,
    Provider,
    Proxy,
    Registry,
    Route,
    load_chains,
    load_env_keys,
    load_keys,
    load_models,
    load_providers,
    load_proxies,
    load_registry,
    load_roles,
    load_turn_limits,
    resolve,
)
from .runner import ModelRunner
from .transport import (
    LOCAL_API_KEY,
    TRANSPORTS,
    ChatRequest,
    ChatResponse,
    LiveStream,
    OpenAIChatTransport,
    Origin,
    StreamBroken,
    StreamChunk,
    ToolCall,
    Transport,
    Usage,
    failover_notes,
    make_transport,
)

__version__ = "0.1.0"

__all__ = [
    # registry
    "AUTH_STYLES",
    "Binding",
    "CHAINS_FILE",
    "Chain",
    "ConfigError",
    "DEFAULT_API_STYLE",
    "DEFAULT_AUTH",
    "DEFAULT_CONTEXT_WINDOW",
    "DEFAULT_PROXY_TYPE",
    "DIRECT_PROXY",
    "DISCOVERED_FILE",
    "ENV_FILE",
    "KEYS_FILE",
    "Key",
    "MODELS_DIR",
    "MODELS_FILE",
    "Model",
    "PROVIDERS_FILE",
    "PROXIES_FILE",
    "Provider",
    "Proxy",
    "ROLE_SUFFIX",
    "ROOT",
    "Registry",
    "Route",
    "TURN_ATTEMPTS_VAR",
    "TURN_DEADLINE_VAR",
    "load_chains",
    "load_env_keys",
    "load_keys",
    "load_models",
    "load_providers",
    "load_proxies",
    "load_registry",
    "load_roles",
    "load_turn_limits",
    "resolve",
    # gateway
    "API_STYLES",
    "Attempt",
    "Executor",
    "Failure",
    "Gateway",
    "GatewayError",
    "LLMError",
    "Rotation",
    "SessionState",
    "TurnBudget",
    "adapter_for",
    "classify",
    "error_detail",
    "provider_error",
    # transport
    "LOCAL_API_KEY",
    "TRANSPORTS",
    "ChatRequest",
    "ChatResponse",
    "LiveStream",
    "OpenAIChatTransport",
    "Origin",
    "StreamBroken",
    "StreamChunk",
    "ToolCall",
    "Transport",
    "Usage",
    "failover_notes",
    "make_transport",
    # runner
    "ModelRunner",
    # discovery
    "Discovered",
    "cache_age",
    "cache_is_fresh",
    "find_extra_providers",
    "load_discovered",
    "merge_into_models",
    "poll_all",
    "poll_provider",
    "print_report",
    "read_cache",
    "render_cache",
    "run_startup_discovery",
    "write_cache",
    # submodules
    "discovery",
    "gateway",
    "registry",
    "runner",
    "transport",
]
