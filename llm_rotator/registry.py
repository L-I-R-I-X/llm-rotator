"""Реестр провайдеров, моделей, ключей и прокси.

Схема хранения — TOML-файлы в подпапке models/ и .env в корне:

    providers.toml  [providers.<имя>]  provider_name, api_style, base_url,
                                        models_url, model_id_prefix, auth,
                                        [providers.<имя>.extra_headers]
    proxies.toml    [proxies.<имя>]    proxy_name, proxy_address, proxy_type,
                                        country, rotates
    keys.toml       [[keys]]           key_name, provider, proxies
    models.toml     [[models]]         model_name, providers, enabled,
                                        prompt_profile, context_window,
                                        max_output_tokens, chat_history_pairs
    chains.toml     [[chains]]         chain_name, models, description
    .env            ключи переменных из keys.toml, роли и пределы хода

`prompt_profile` и `chat_history_pairs` ротация читает, но не толкует: это
непрозрачные поля для вызывающего кода (сборка личности, окно истории).

Модель идентифицируется только по имени: одна запись на model_name, сколько бы
провайдеров её ни отдавали. Провайдеры перечислены у модели в порядке
предпочтения, и это же порядок фейловера.

Цепочка (chains.toml) — это порядок моделей целиком: сначала все провайдеры
первой, потом второй и так далее. Роль из .env ссылается на цепочку по имени,
поэтому код знает только имя роли: `PERSONA_MODELCHAIN = "persona"` говорит,
что роль persona работает на цепочке persona. Появятся роль под агента или под
память — добавятся цепочка и строка в .env, перебор кода не меняется.

Ключ принадлежит ровно одному провайдеру и ходит через собственный список
прокси. Значения ключей лежат только в .env; здесь имена переменных
превращаются в готовые связки (Binding) через Route.

Единица обхода — связка (ключ, прокси), см. llm_gateway. Связку нельзя разобрать
на части: ключ не ходит через чужой адрес, поэтому несколько аккаунтов с одного
IP не получаются.

Провайдеры без авторизации (локальный ollama и подобные) объявляются полем
`auth = "none"`: им ключ в .env не нужен, и запись в keys.toml нужна только для
связки с адресом. Всё остальное — по умолчанию `auth = "bearer"`.

Валидация ссылочной целостности выполняется на загрузке и падает сразу, до
первого сетевого запроса, с перечислением допустимых имён.
"""

from __future__ import annotations

import tomllib
from collections.abc import Iterable, Iterator
from dataclasses import dataclass
from pathlib import Path

from dotenv import dotenv_values

ROOT = Path.cwd()
"""База конфигурации по умолчанию: каталог, откуда запущен процесс.

Пакет не привязан к собственной директории: файлы конкретного проекта лежат
рядом с его кодом, а не рядом с библиотекой. Кто знает свой корень, передаёт его
явно в ``load_registry(base_dir=...)``.
"""

MODELS_DIR = Path("models")
PROVIDERS_FILE = MODELS_DIR / "providers.toml"
MODELS_FILE = MODELS_DIR / "models.toml"
PROXIES_FILE = MODELS_DIR / "proxies.toml"
KEYS_FILE = MODELS_DIR / "keys.toml"
CHAINS_FILE = MODELS_DIR / "chains.toml"
DISCOVERED_FILE = MODELS_DIR / "models.discovered.toml"
ENV_FILE = Path(".env")

DIRECT_PROXY = "direct"

ROLE_SUFFIX = "_MODELCHAIN"
"""Хвост переменной .env, за которой стоит имя роли.

`PERSONA_MODELCHAIN = "persona"` — роль persona работает на цепочке persona.
Роль называет код, цепочку выбирает конфиг, поэтому новые роли не требуют
правок в коде.
"""

TURN_DEADLINE_VAR = "LLM_TURN_DEADLINE"
TURN_ATTEMPTS_VAR = "LLM_TURN_MAX_ATTEMPTS"
"""Пределы одного хода в .env: время в секундах и число попыток.

Считаются на весь ход целиком, на все модели цепочки: без них три модели по
шесть адресов — это до 72 попыток, и ход может не вернуться.
"""

DEFAULT_API_STYLE = "openai"
DEFAULT_AUTH = "bearer"

AUTH_STYLES: tuple[str, ...] = ("bearer", "none")
"""Как провайдер принимает авторизацию.

`bearer` — обычный `Authorization: Bearer <ключ>`, ключ обязан быть в .env.
`none`   — авторизации нет вовсе (локальный ollama, свой сервер): значение
           ключа не читается, а запись в keys.toml нужна только ради адресов.
"""

DEFAULT_PROXY_TYPE = "http"
PROXY_TYPES: tuple[str, ...] = ("http", "https", "socks5", "socks5h")
"""Схема адреса прокси, которой подключается httpx.

Значение попадает в `proxy_address`, если схема там не написана: `127.0.0.1:1091`
с `proxy_type = "socks5"` становится `socks5://127.0.0.1:1091`. Разница между
`socks5` и `socks5h` — где резолвится имя прокси: локально или на стороне
прокси-сервера, что важно, когда прокси задан именем, а не адресом.

Для socks5 нужен пакет socksio, он ставится вместе с httpx: `httpx[socks]`.
"""

DEFAULT_CONTEXT_WINDOW = 32768
"""Окно контекста по умолчанию, если модель не объявила своё (context_window).

Число сообщений, которые влезут в запрос, считают от него: память между
диалогами будет класть в контекст последние пары и сводку от маленькой модели.
"""


def resolve(path: Path, base: Path | None = None) -> Path:
    """Относительный путь — от базового каталога, абсолютный не трогаем.

    По умолчанию база — текущая рабочая директория. Вызывающий код, у которого
    конфигурация лежит рядом с кодом, а не там, откуда запущен процесс, обязан
    передать её явно (``load_registry(base_dir=...)``): пакет ничего не знает о
    том, где лежат файлы конкретного проекта.
    """
    if path.is_absolute():
        return path
    return (base if base is not None else Path.cwd()) / path


class ConfigError(Exception):
    """Некорректная конфигурация реестра."""


@dataclass(frozen=True, slots=True)
class Proxy:
    """Адрес, через который можно достучаться до API."""

    name: str
    address: str
    country: str = ""
    rotates: bool = False
    proxy_type: str = DEFAULT_PROXY_TYPE

    @property
    def is_direct(self) -> bool:
        return self.name == DIRECT_PROXY

    @property
    def is_socks(self) -> bool:
        return self.proxy_type.startswith("socks")

    @property
    def httpx_proxy(self) -> str | None:
        """Адрес для httpx: со схемой, иначе библиотека его не примет.

        Пустая строка (прямое соединение) — это None: у httpx параметр proxy
        по умолчанию и есть «без прокси», а пустая строка означала бы попытку
        подключиться к пустому адресу.
        """
        if not self.address:
            return None
        if "://" in self.address:
            return self.address
        return f"{self.proxy_type}://{self.address}"


@dataclass(frozen=True, slots=True)
class Key:
    """Ключ API: имя переменной в .env, провайдер и разрешённые адреса."""

    name: str
    provider: str
    proxies: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class Provider:
    """Провайдер API ИИ-моделей."""

    name: str
    api_style: str
    base_url: str
    models_url: str
    model_id_prefix: str = ""
    auth: str = DEFAULT_AUTH
    extra_headers: tuple[tuple[str, str], ...] = ()

    @property
    def requires_key(self) -> bool:
        """Нужно ли значение ключа в .env для этого провайдера."""
        return self.auth != "none"

    @property
    def headers(self) -> dict[str, str]:
        return dict(self.extra_headers)

    def chat_url(self, path: str = "chat/completions") -> str:
        return f"{self.base_url.rstrip('/')}/{path.lstrip('/')}"

    def strip_id_prefix(self, name: str) -> str:
        """Убрать префикс ресурса из имени модели.

        Нужен Google: его OpenAI-совместимый /models отдаёт имена как
        models/gemini-3.8-flash, а /chat/completions ждёт gemini-3.8-flash.
        """
        return name.removeprefix(self.model_id_prefix) if self.model_id_prefix else name


@dataclass(frozen=True, slots=True)
class Model:
    """Модель реестра. Идентичность — только имя."""

    name: str
    providers: tuple[str, ...]
    enabled: bool = True
    prompt_profile: str = ""
    """Имя профиля промпта из models.toml. Значение не проверяется: какие имена
    допустимы, знает вызывающий код (сборка личности), а не ротация."""
    context_window: int = DEFAULT_CONTEXT_WINDOW
    max_output_tokens: int | None = None
    chat_history_pairs: int | None = None
    """Размер окна истории для модели из models.toml. Непрозрачное для ротации
    поле: политику (модель → .env → дефолт) решает вызывающий код."""
    """Потолок длины ответа, который уходит провайдеру как max_tokens.

    Нужен не для краткости, а чтобы пройти проверку провайдера: Groq сверяет
    заявленный потолок с лимитом выходных токенов в минуту (OTPM) и отклоняет
    запрос целиком, если потолок выше лимита. Без max_tokens провайдер берёт
    свой дефолт, и на длинной переписке он упирается в лимит даже на коротком
    вопросе. None = поле не отправляется, провайдер решает сам.
    """

    @property
    def label(self) -> str:
        return self.name


@dataclass(frozen=True, slots=True)
class Chain:
    """Именованный порядок моделей: кто отвечает первым, кто следующим.

    Роль из .env ссылается на цепочку по имени, поэтому код не знает ни одного
    имени намертво: появится роль под агента или под память — добавятся цепочка
    и строка в .env, а перебор кода не меняется.
    """

    name: str
    models: tuple[Model, ...]
    description: str = ""

    @property
    def model_names(self) -> tuple[str, ...]:
        return tuple(model.name for model in self.models)

    def __str__(self) -> str:
        return " → ".join(self.model_names)


@dataclass(frozen=True, slots=True)
class Binding:
    """Неразбираемая связка: конкретный ключ через конкретный адрес."""

    key_name: str
    key_value: str
    proxy: Proxy

    def __str__(self) -> str:
        return f"{self.key_name}@{self.proxy.name}"


@dataclass(frozen=True, slots=True)
class Route:
    """Одна модель у одного провайдера, развёрнутая в упорядоченные связки."""

    model: Model
    provider: Provider
    bindings: tuple[Binding, ...]

    @property
    def label(self) -> str:
        return f"{self.model.name} @ {self.provider.name}"

    @property
    def key_names(self) -> tuple[str, ...]:
        seen: list[str] = []
        for binding in self.bindings:
            if binding.key_name not in seen:
                seen.append(binding.key_name)
        return tuple(seen)

    @property
    def proxy_names(self) -> tuple[str, ...]:
        seen: list[str] = []
        for binding in self.bindings:
            if binding.proxy.name not in seen:
                seen.append(binding.proxy.name)
        return tuple(seen)

    def __str__(self) -> str:
        pairs = ", ".join(str(b) for b in self.bindings) or "-"
        return f"{self.label} [{pairs}]"


class Registry:
    """Загруженный и провалидированный набор конфигурации."""

    def __init__(
        self,
        providers: dict[str, Provider],
        proxies: dict[str, Proxy],
        keys: dict[str, Key],
        models: dict[str, Model],
        env_keys: dict[str, str],
        warnings: list[str] | None = None,
        chains: dict[str, Chain] | None = None,
        roles: dict[str, str] | None = None,
        turn_deadline: float | None = None,
        turn_max_attempts: int | None = None,
    ) -> None:
        self.providers = providers
        self.proxies = proxies
        self.keys = keys
        self.models = models
        self.env_keys = env_keys
        self.warnings = warnings or []
        self.chains = chains or {}
        self.roles = roles or {}
        self.turn_deadline = turn_deadline
        self.turn_max_attempts = turn_max_attempts

    def provider(self, name: str) -> Provider:
        try:
            return self.providers[name]
        except KeyError:
            raise ConfigError(
                f"Провайдер {name!r} не найден. Доступные: {_fmt_names(self.providers)}"
            ) from None

    def proxy(self, name: str) -> Proxy:
        try:
            return self.proxies[name]
        except KeyError:
            raise ConfigError(
                f"Прокси {name!r} не найден. Доступные: {_fmt_names(self.proxies)}"
            ) from None

    def key(self, name: str) -> Key:
        try:
            return self.keys[name]
        except KeyError:
            raise ConfigError(
                f"Ключ {name!r} не найден. Доступные: {_fmt_names(self.keys)}"
            ) from None

    def model(self, name: str) -> Model:
        try:
            return self.models[name]
        except KeyError:
            close = sorted(n for n in self.models if n.startswith(name))
            hint = f" Похожие: {_fmt_names(close)}." if close else ""
            raise ConfigError(f"Модель {name!r} не найдена.{hint}") from None

    def chain(self, name: str) -> Chain:
        try:
            return self.chains[name]
        except KeyError:
            close = sorted(n for n in self.chains if n.startswith(name))
            hint = f" Похожие: {_fmt_names(close)}." if close else ""
            raise ConfigError(f"Цепочка {name!r} не найдена.{hint}") from None

    def chain_for_role(self, role: str) -> Chain:
        """Цепочка, назначенная роли в .env.

        Имя цепочки в коде не зашито: роль приходит из кода, цепочка — из .env.
        """
        try:
            chain_name = self.roles[role]
        except KeyError:
            roles = ", ".join(f"{r} → {c}" for r, c in self.roles.items())
            raise ConfigError(
                f"Роль {role!r} не описана в .env. Нужна переменная "
                f'{role.upper()}{ROLE_SUFFIX} = "<цепочка>"'
                + (f". Описаны: {roles}." if roles else " Сейчас ролей нет.")
            ) from None
        return self.chain(chain_name)

    def roles_for_chain(self, chain_name: str) -> tuple[str, ...]:
        """Какие роли работают на этой цепочке — для /models и проверок."""
        return tuple(r for r, c in self.roles.items() if c == chain_name)

    def chain_routes(self, role: str) -> list[list[Route]]:
        """Маршруты цепочки по моделям: список списков, в порядке моделей.

        Обход делает Gateway.rotate, поэтому каждая модель обходится отдельно и
        между ними можно спросить пользователя. Так и устроен переход: сначала
        все провайдеры первой модели, потом всех второй и так далее.
        """
        chain = self.chain_for_role(role)
        return [self.routes(model.name) for model in chain.models]

    def keys_for(self, provider_name: str) -> list[Key]:
        """Ключи провайдера в порядке keys.toml."""
        return [k for k in self.keys.values() if k.provider == provider_name]

    def enabled_models(self) -> Iterator[Model]:
        for model in self.models.values():
            if model.enabled:
                yield model

    def bindings_for(self, provider_name: str) -> tuple[Binding, ...]:
        """Все рабочие связки провайдера: ключ снаружи, его прокси внутри.

        У провайдера с auth = "none" значение ключа не читается из .env вовсе,
        связка всё равно строится: она нужна, чтобы закрепить за провайдером
        адрес.
        """
        bindings: list[Binding] = []
        for key in self.keys_for(provider_name):
            value = self.env_keys.get(key.name) or ""
            if not value and self.provider(key.provider).requires_key:
                continue
            for proxy_name in key.proxies:
                bindings.append(
                    Binding(
                        key_name=key.name,
                        key_value=value,
                        proxy=self.proxy(proxy_name),
                    )
                )
        return tuple(bindings)

    def credential_problem(self, provider_name: str) -> str:
        """Почему у провайдера нет ни одной связки."""
        provider = self.provider(provider_name)
        if not provider.requires_key:
            return "у него не заведено ни одного ключа в keys.toml"
        without_key = [
            k.name
            for k in self.keys_for(provider_name)
            if not self.env_keys.get(k.name)
        ]
        if without_key:
            return f"в .env нет {_fmt_names(without_key)}"
        return "у него не заведено ни одного ключа в keys.toml"

    def route(self, model_name: str, provider_name: str) -> Route:
        """Связки (ключ, прокси) для модели у провайдера, в порядке обхода.

        Ключ снаружи, его прокси внутри: смена адреса не съедает ключ, а смена
        ключа начинается с первого адреса следующего ключа.
        """
        model = self.model(model_name)
        if not model.enabled:
            raise ConfigError(f"Модель {model.label} отключена (enabled = false)")

        provider = self.provider(provider_name)
        bindings = self.bindings_for(provider_name)
        if not bindings:
            raise ConfigError(
                f"{model.label} @ {provider.name}: нечем выполнять запрос — "
                f"{self.credential_problem(provider_name)}"
            )
        return Route(model=model, provider=provider, bindings=bindings)

    def routes(self, model_name: str) -> list[Route]:
        """Маршруты по всем провайдерам модели в порядке предпочтения."""
        model = self.model(model_name)
        return [self.route(model.name, name) for name in model.providers]

    def all_routes(self) -> Iterator[Route]:
        for model in self.enabled_models():
            for provider_name in model.providers:
                yield self.route(model.name, provider_name)

    def provider_routes(self, provider_name: str) -> list[Route]:
        """Все включённые модели провайдера — для отчётов и проверок."""
        return [
            self.route(m.name, provider_name)
            for m in self.enabled_models()
            if provider_name in m.providers
        ]


def _fmt_names(names: Iterable[str]) -> str:
    items = sorted(names)
    return ", ".join(repr(n) for n in items) if items else "(пусто)"


def _load_toml(path: Path) -> dict:
    try:
        with path.open("rb") as fh:
            return tomllib.load(fh)
    except FileNotFoundError:
        raise ConfigError(f"Файл не найден: {path}") from None
    except tomllib.TOMLDecodeError as exc:
        raise ConfigError(f"{path.name}: не удалось разобрать TOML — {exc}") from None


def _table(doc: dict, key: str, filename: str) -> dict:
    raw = doc.get(key)
    if not isinstance(raw, dict):
        raise ConfigError(f"{filename}: ожидается секция [{key}.*]")
    return raw


def _array(doc: dict, key: str, filename: str) -> list:
    """Массив таблиц верхнего уровня. Отсутствие блока — пустой список."""
    raw = doc.get(key)
    if raw is None:
        return []
    if not isinstance(raw, list):
        raise ConfigError(f"{filename}: ожидается массив таблиц [[{key}]]")
    return raw


def _req_str(table: dict, field: str, where: str) -> str:
    value = table.get(field)
    if not isinstance(value, str) or not value.strip():
        raise ConfigError(f"{where}: поле {field!r} обязательно и должно быть строкой")
    return value.strip()


def _opt_str(table: dict, field: str, where: str, default: str = "") -> str:
    value = table.get(field, default)
    if not isinstance(value, str):
        raise ConfigError(f"{where}: поле {field!r} должно быть строкой")
    return value.strip()


def _opt_bool(table: dict, field: str, where: str, default: bool) -> bool:
    value = table.get(field, default)
    if not isinstance(value, bool):
        raise ConfigError(f"{where}: поле {field!r} должно быть true или false")
    return value


def _opt_limit(table: dict, field: str, where: str) -> int | None:
    """Необязательный положительный потолок: нет поля — нет значения."""
    if field not in table:
        return None
    return _opt_int(table, field, where, 1)


def _opt_int(table: dict, field: str, where: str, default: int) -> int:
    value = table.get(field, default)
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise ConfigError(f"{where}: поле {field!r} должно быть положительным числом")
    return value


def _opt_int_or_none(table: dict, field: str, where: str) -> int | None:
    if field not in table:
        return None
    value = table[field]
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise ConfigError(f"{where}: поле {field!r} должно быть положительным числом")
    return value


def _name_list(table: dict, field: str, where: str) -> tuple[str, ...]:
    value = table.get(field)
    if not isinstance(value, list) or not value:
        raise ConfigError(f"{where}: поле {field!r} должно быть непустым списком строк")
    out: list[str] = []
    for item in value:
        if not isinstance(item, str) or not item.strip():
            raise ConfigError(f"{where}: поле {field!r} содержит не-строку: {item!r}")
        out.append(item.strip())
    if len(set(out)) != len(out):
        raise ConfigError(f"{where}: поле {field!r} содержит повторяющиеся имена")
    return tuple(out)


def _extra_headers(table: dict, where: str) -> tuple[tuple[str, str], ...]:
    """Заголовки, которые провайдер требует сверх авторизации.

    Нужны, например, OpenRouter (HTTP-Referer, X-Title) и любому провайдеру с
    собственной валютой или репутацией. Порядок сохраняется: заголовки уходят
    в запрос как есть.
    """
    raw = table.get("extra_headers", {})
    if not isinstance(raw, dict):
        raise ConfigError(
            f"{where}: секция 'extra_headers' должна быть таблицей вида "
            '"имя_заголовка" = "значение"'
        )
    out: list[tuple[str, str]] = []
    for header, value in raw.items():
        if not isinstance(value, str):
            raise ConfigError(
                f"{where} [extra_headers]: значение заголовка {header!r} должно "
                "быть строкой"
            )
        try:
            value.encode("ascii")
        except UnicodeEncodeError:
            raise ConfigError(
                f"{where} [extra_headers]: значение заголовка {header!r} содержит "
                "не-ASCII символы — HTTP-заголовки такими не отправляются"
            ) from None
        out.append((header, value))
    return tuple(out)


def _check_name_match(table_key: str, declared: str, where: str) -> None:
    if table_key != declared:
        raise ConfigError(
            f"{where}: поле с именем {declared!r} не совпадает с ключом таблицы "
            f"[{table_key!r}] — опечатка или недосинхронизированный файл"
        )


def load_proxies(path: Path = PROXIES_FILE) -> dict[str, Proxy]:
    path = resolve(path)
    doc = _load_toml(path)
    raw = _table(doc, "proxies", path.name)
    out: dict[str, Proxy] = {}
    for name, table in raw.items():
        where = f"{path.name} [proxies.{name}]"
        if not isinstance(table, dict):
            raise ConfigError(f"{where}: ожидается таблица")
        declared = _req_str(table, "proxy_name", where)
        _check_name_match(name, declared, where)
        address = _opt_str(table, "proxy_address", where)
        if not isinstance(table.get("proxy_address"), str):
            raise ConfigError(
                f"{where}: поле 'proxy_address' обязательно и должно быть строкой"
            )
        proxy_type = _opt_str(table, "proxy_type", where, DEFAULT_PROXY_TYPE)
        if proxy_type not in PROXY_TYPES:
            raise ConfigError(
                f"{where}: поле 'proxy_type' = {proxy_type!r} не поддерживается. "
                f"Допустимо: {_fmt_names(PROXY_TYPES)}"
            )
        if address and "://" in address:
            written = address.split("://", 1)[0]
            if written not in PROXY_TYPES:
                raise ConfigError(
                    f"{where}: в proxy_address схема {written!r} не поддерживается. "
                    f"Допустимо: {_fmt_names(PROXY_TYPES)}"
                )
            if proxy_type != DEFAULT_PROXY_TYPE and written != proxy_type:
                raise ConfigError(
                    f"{where}: proxy_address сам задаёт схему {written!r}, а "
                    f"proxy_type = {proxy_type!r} — противоречие. Убери одно из двух"
                )
            proxy_type = written
        out[name] = Proxy(
            name=name,
            address=address,
            country=_opt_str(table, "country", where),
            rotates=_opt_bool(table, "rotates", where, False),
            proxy_type=proxy_type,
        )

    if DIRECT_PROXY not in out:
        raise ConfigError(
            f"{path.name}: отсутствует обязательный прокси-заглушка "
            f"[proxies.{DIRECT_PROXY}] с пустым proxy_address"
        )
    if out[DIRECT_PROXY].address:
        raise ConfigError(
            f"{path.name}: прокси {DIRECT_PROXY!r} обязан иметь пустой proxy_address "
            "(прямое соединение)"
        )
    return out


def load_providers(path: Path = PROVIDERS_FILE) -> dict[str, Provider]:
    path = resolve(path)
    doc = _load_toml(path)
    raw = _table(doc, "providers", path.name)
    out: dict[str, Provider] = {}
    for name, table in raw.items():
        where = f"{path.name} [providers.{name}]"
        if not isinstance(table, dict):
            raise ConfigError(f"{where}: ожидается таблица")
        declared = _req_str(table, "provider_name", where)
        _check_name_match(name, declared, where)
        api_style = _opt_str(table, "api_style", where, DEFAULT_API_STYLE)
        if not api_style:
            raise ConfigError(f"{where}: поле 'api_style' не может быть пустым")
        auth = _opt_str(table, "auth", where, DEFAULT_AUTH)
        if auth not in AUTH_STYLES:
            raise ConfigError(
                f"{where}: поле 'auth' = {auth!r} не поддерживается. "
                f"Допустимо: {_fmt_names(AUTH_STYLES)}"
            )
        out[name] = Provider(
            name=name,
            api_style=api_style,
            base_url=_req_str(table, "base_url", where),
            models_url=_req_str(table, "models_url", where),
            model_id_prefix=_opt_str(table, "model_id_prefix", where),
            auth=auth,
            extra_headers=_extra_headers(table, where),
        )
    if not out:
        raise ConfigError(f"{path.name}: не описан ни один провайдер")
    return out


def load_keys(
    path: Path = KEYS_FILE,
    proxies: dict[str, Proxy] | None = None,
    providers: dict[str, Provider] | None = None,
) -> dict[str, Key]:
    path = resolve(path)
    if proxies is None:
        proxies = load_proxies()
    if providers is None:
        providers = load_providers()

    doc = _load_toml(path)
    raw = _array(doc, "keys", path.name)
    out: dict[str, Key] = {}
    for index, table in enumerate(raw, start=1):
        where = f"{path.name} [[keys]] #{index}"
        if not isinstance(table, dict):
            raise ConfigError(f"{where}: ожидается таблица")
        name = _req_str(table, "key_name", where)
        provider = _req_str(table, "provider", where)
        where = f"{path.name} [{name}]"
        if name in out:
            raise ConfigError(
                f"{where}: дубликат key_name — ключ принадлежит одному провайдеру"
            )
        if provider not in providers:
            raise ConfigError(
                f"{where}: провайдер {provider!r} не найден в providers.toml. "
                f"Доступные: {_fmt_names(providers)}"
            )
        proxy_names = _name_list(table, "proxies", where)
        for pname in proxy_names:
            if pname not in proxies:
                raise ConfigError(
                    f"{where}: прокси {pname!r} не найден в proxies.toml. "
                    f"Доступные: {_fmt_names(proxies)}"
                )
        out[name] = Key(name=name, provider=provider, proxies=proxy_names)
    return out


def load_models(
    path: Path = MODELS_FILE, providers: dict[str, Provider] | None = None
) -> dict[str, Model]:
    path = resolve(path)
    doc = _load_toml(path)
    raw = _array(doc, "models", path.name)
    if providers is None:
        providers = load_providers()

    out: dict[str, Model] = {}
    for index, table in enumerate(raw, start=1):
        where = f"{path.name} [[models]] #{index}"
        if not isinstance(table, dict):
            raise ConfigError(f"{where}: ожидается таблица")
        name = _req_str(table, "model_name", where)
        where = f"{path.name} [{name}]"
        provider_names = _name_list(table, "providers", where)
        for pname in provider_names:
            if pname not in providers:
                raise ConfigError(
                    f"{where}: провайдер {pname!r} не найден в providers.toml. "
                    f"Доступные: {_fmt_names(providers)}"
                )
        if name in out:
            raise ConfigError(
                f"{where}: дубликат — модель идентифицируется только по имени, "
                "опиши её один раз"
            )
        out[name] = Model(
            name=name,
            providers=provider_names,
            enabled=_opt_bool(table, "enabled", where, True),
            prompt_profile=_opt_str(table, "prompt_profile", where, ""),
            context_window=_opt_int(
                table, "context_window", where, DEFAULT_CONTEXT_WINDOW
            ),
            max_output_tokens=_opt_limit(table, "max_output_tokens", where),
            chat_history_pairs=_opt_int_or_none(table, "chat_history_pairs", where),
        )
    return out


def load_chains(
    path: Path = CHAINS_FILE, models: dict[str, Model] | None = None
) -> dict[str, Chain]:
    """Загрузить именованные цепочки моделей и проверить каждую запись."""
    path = resolve(path)
    if not path.exists():
        return {}
    doc = _load_toml(path)
    raw = _array(doc, "chains", path.name)

    out: dict[str, Chain] = {}
    for index, table in enumerate(raw, start=1):
        where = f"{path.name} [[chains]] #{index}"
        if not isinstance(table, dict):
            raise ConfigError(f"{where}: ожидается таблица")
        name = _req_str(table, "chain_name", where)
        where = f"{path.name} [{name}]"
        if name in out:
            raise ConfigError(f"{where}: дубликат — цепочка описана дважды")

        if models is None:
            models = load_models()
        picked: list[Model] = []
        for mname in _name_list(table, "models", where):
            model = models.get(mname)
            if model is None:
                close = sorted(n for n in models if n.startswith(mname))
                hint = f" Похожие: {_fmt_names(close)}." if close else ""
                raise ConfigError(
                    f"{where}: модель {mname!r} не найдена в models.toml.{hint}"
                )
            if model in picked:
                raise ConfigError(
                    f"{where}: модель {mname!r} повторяется — в цепочке она нужна "
                    "один раз, порядок и так задан"
                )
            if not model.enabled:
                raise ConfigError(
                    f"{where}: модель {mname!r} выключена (enabled = false) и не "
                    "может отвечать: уберите её из цепочки или включите в models.toml"
                )
            picked.append(model)
        out[name] = Chain(
            name=name,
            models=tuple(picked),
            description=_opt_str(table, "description", where, ""),
        )
    return out


def load_roles(env: dict[str, str], chains: dict[str, Chain]) -> dict[str, str]:
    """Роли из .env: PERSONA_MODELCHAIN = "persona" → роль persona на цепочку.

    Имя роли живёт в коде (`client.conversation(role="persona")`), а цепочка под
    ней выбирается конфигом. Появятся новые роли — добавятся строки сюда.
    """
    roles: dict[str, str] = {}
    for name, value in env.items():
        upper = name.upper()
        if not upper.endswith(ROLE_SUFFIX):
            continue
        role = upper[: -len(ROLE_SUFFIX)].lower()
        if not role:
            raise ConfigError(
                f"{ENV_FILE}: переменная {name!r} без имени роли, нужно "
                f'например PERSONA{ROLE_SUFFIX} = "persona"'
            )
        if value not in chains:
            raise ConfigError(
                f"{ENV_FILE}: роль {role!r} ссылается на цепочку {value!r}, которой "
                f"нет в chains.toml. Доступные цепочки: {_fmt_names(chains)}"
            )
        roles[role] = value
    return roles


def load_turn_limits(env: dict[str, str]) -> tuple[float | None, int | None]:
    """Пределы хода из .env: время в секундах и число попыток."""
    deadline = _env_float(env, TURN_DEADLINE_VAR)
    attempts = _env_int(env, TURN_ATTEMPTS_VAR)
    return deadline, attempts


def _env_float(env: dict[str, str], name: str) -> float | None:
    raw = env.get(name)
    if raw is None:
        return None
    try:
        value = float(raw)
    except ValueError:
        raise ConfigError(
            f"{ENV_FILE}: {name} = {raw!r} — нужно число секунд, например {name} = 150"
        ) from None
    if value <= 0:
        raise ConfigError(f"{ENV_FILE}: {name} = {raw!r} должно быть больше нуля")
    return value


def _env_int(env: dict[str, str], name: str) -> int | None:
    raw = env.get(name)
    if raw is None:
        return None
    try:
        value = int(raw)
    except ValueError:
        raise ConfigError(
            f"{ENV_FILE}: {name} = {raw!r} — нужно целое число, например {name} = 8"
        ) from None
    if value <= 0:
        raise ConfigError(f"{ENV_FILE}: {name} = {raw!r} должно быть больше нуля")
    return value


def load_env_keys(path: Path = ENV_FILE) -> dict[str, str]:
    path = resolve(path)
    if not path.exists():
        raise ConfigError(f"Файл с ключами не найден: {path}")
    values = dotenv_values(path)
    return {name: value.strip() for name, value in values.items() if value}


def _shared_proxy_warnings(
    keys: dict[str, Key], proxies: dict[str, Proxy]
) -> list[str]:
    """Адрес, забитый под несколько ключей, — риск абуза на бесплатных лимитах."""
    by_proxy: dict[str, list[str]] = {}
    for key in keys.values():
        for pname in key.proxies:
            by_proxy.setdefault(pname, []).append(key.name)

    warnings: list[str] = []
    for pname, holders in sorted(by_proxy.items()):
        if len(holders) < 2:
            continue
        proxy = proxies[pname]
        detail = ", ".join(sorted(holders))
        if proxy.rotates:
            warnings.append(
                f"прокси {pname!r} ({proxy.country or 'без страны'}) привязан к "
                f"ключам {detail} — у него rotates = true, разные выходные IP, "
                "так что несколько аккаунтов законны"
            )
        elif proxy.is_direct:
            warnings.append(
                f"прокси {pname!r} — это прямое соединение, и оно общее у ключей "
                f"{detail}: все эти аккаунты идут с одного IP. Если у провайдера "
                "бесплатные лимиты, заведи каждому ключу свой адрес"
            )
        else:
            warnings.append(
                f"прокси {pname!r} ({proxy.country or 'без страны'}) привязан к "
                f"нескольким ключам: {detail}. Если у провайдера бесплатные "
                "лимиты, это выглядит как абуз нескольких аккаунтов с одного IP — "
                "разведи ключи по разным адресам либо поставь rotates = true"
            )
    return warnings


def _unused_proxy_warnings(
    keys: dict[str, Key], proxies: dict[str, Proxy]
) -> list[str]:
    used = {pname for key in keys.values() for pname in key.proxies}
    unused = sorted(p for p in proxies if p not in used)
    if not unused:
        return []
    return [
        f"прокси {', '.join(repr(p) for p in unused)} не привязан ни к одному ключу"
    ]


def _validate_credentials(registry: Registry) -> list[str]:
    """Сверить ключи с .env. Ошибка — если провайдер нужен включённой модели."""
    warnings: list[str] = []
    for name, key in registry.keys.items():
        if not registry.provider(key.provider).requires_key:
            continue
        if registry.env_keys.get(key.name):
            continue
        users = sorted(
            m.label
            for m in registry.models.values()
            if key.provider in m.providers and m.enabled
        )
        detail = f"ключ {name!r} для провайдера {key.provider!r} не найден в .env"
        if users:
            raise ConfigError(
                f"{detail}, а он нужен моделям: {', '.join(users)}. "
                "Добавь переменную в .env или убери провайдера из их списка"
            )
        warnings.append(f"{detail} — провайдер не используется включёнными моделями")

    for provider_name in registry.providers:
        if not any(key.provider == provider_name for key in registry.keys.values()):
            warnings.append(
                f"у провайдера {provider_name!r} нет ни одного ключа в keys.toml — "
                "он не будет использоваться"
            )
    return warnings


def load_registry(base_dir: Path | None = None) -> Registry:
    """Загрузить и провалидировать все файлы конфигурации вместе с .env.

    `base_dir` — каталог, в котором лежат `models/` и `.env`. Без него берётся
    текущая рабочая директория (``ROOT``).
    """
    base = ROOT if base_dir is None else base_dir
    proxies = load_proxies(base / PROXIES_FILE)
    providers = load_providers(base / PROVIDERS_FILE)
    keys = load_keys(base / KEYS_FILE, proxies=proxies, providers=providers)
    models = load_models(base / MODELS_FILE, providers=providers)
    chains = load_chains(base / CHAINS_FILE, models=models)
    env_keys = load_env_keys(base / ENV_FILE)
    roles = load_roles(env_keys, chains)
    turn_deadline, turn_max_attempts = load_turn_limits(env_keys)

    registry = Registry(
        providers=providers,
        proxies=proxies,
        keys=keys,
        models=models,
        env_keys=env_keys,
        chains=chains,
        roles=roles,
        turn_deadline=turn_deadline,
        turn_max_attempts=turn_max_attempts,
    )
    warnings = _validate_credentials(registry)
    warnings += _shared_proxy_warnings(keys, proxies)
    warnings += _unused_proxy_warnings(keys, proxies)
    warnings += _chain_warnings(registry)
    registry.warnings = warnings
    return registry


def _chain_warnings(registry: Registry) -> list[str]:
    """Модель вне цепочек и цепочка без ролей — типичная забывчивость."""
    if not registry.chains:
        return []  # цепочек нет — работа идёт по одной модели, ругаться не на что
    warnings: list[str] = []
    used = {name for chain in registry.chains.values() for name in chain.model_names}
    for model in registry.enabled_models():
        if model.name not in used:
            warnings.append(
                f"модель {model.name!r} не входит ни в одну цепочку: в диалоге "
                "она не понадобится, пока её не добавят в chains.toml"
            )
    for name in registry.chains:
        if not registry.roles_for_chain(name):
            warnings.append(
                f"цепочка {name!r} никем не используется: в .env нет переменной "
                f'*{ROLE_SUFFIX} = "{name}"'
            )
    return warnings


if __name__ == "__main__":
    try:
        reg = load_registry()
    except ConfigError as exc:
        raise SystemExit(f"Ошибка конфигурации: {exc}") from None
    for warning in reg.warnings:
        print(f"ВНИМАНИЕ: {warning}")
    print(
        f"Провайдеров: {len(reg.providers)}  "
        f"ключей: {len(reg.keys)}  "
        f"прокси: {len(reg.proxies)}  "
        f"моделей: {len(reg.models)} "
        f"(включено {sum(1 for _ in reg.enabled_models())})"
    )
    print()
    for route in reg.all_routes():
        print(f"  {route}")
