"""Обход провайдеров, ключей и адресов — общий для всех способов отправки.

Единица обхода — связка (ключ, прокси), см. model_registry.Binding. Связка
неразбираема: ключ ходит только через адреса из своего списка, поэтому два
аккаунта с одного IP не получаются. Никакого декартова произведения
прокси × ключ нет и быть не должно.

Сам цикл обхода не знает, как именно уходит запрос: он просит у вызывающего
исполнителя попытки (Executor) сделать одну попытку и говорит, чем она
кончилась, а разбираться с этим — единственное дело исполнителя
(Failure). На этом же цикле построены и сырой httpx-вызов (request_routes,
им ходят list_models и discovery), и вызовы чата через openai-совместимый
слой, см. llm_transport. Правила отказа, cooldown и запрет картова
произведения живут здесь в одном месте и не дублируются под каждый API.

Обход: провайдеры модели в порядке её списка providers → внутри провайдера
связки в порядке ключей, а внутри ключа его прокси. Ключ снаружи, адрес внутри:
смена адреса не съедает ключ. На каждой неудаче решается, что именно сломалось:

    401 / 402 / 403      →  сломан ключ    → следующий ключ
    404                  →  провайдер не подходит → следующий провайдер
    403 про квоту / 429  →  ключ жив, пауза → следующий ключ
    429 «request too large» → запрос велик → следующий провайдер модели
    502 / 503 / 504      →  сломан адрес  → следующий адрес тем же ключом
    таймаут / обрыв      →  сломан адрес  → следующий адрес тем же ключом
    прочие 5xx           →  модель недоступна → следующий провайдер модели
    400 / 422            →  запрос неверен → перебор бессмыслен, ошибка наверх

Обход ограничен TurnBudget: временем и числом попыток на весь ход, общих для
всех моделей цепочки. Без предела три модели по шесть адресов — это до 72
попыток, и ход может висеть очень долго.

Пять уточнений, продиктованных реальными ответами провайдеров:

* 401/403 без структурированной ошибки провайдера (тело не JSON или не содержит
  поля error) — это не ключ, а блок посредника: Cloudflare/WAF/прокси. Такой
  ответ означает заблокированный адрес, поэтому считается сломанным адресом.
  Проверено на живых ключах: Groq отдаёт 200 на /models и 403 "error code: 1010"
  на /chat/completions — это бан адреса Cloudflare, а не ключа.

* 429 и 403 с текстом про квоту — это временное исчерпание лимита, а не мёртвый
  ключ: он уходит не в bad_keys, а на паузу, и возвращается в ротацию по
  истечении паузы. Если провайдер прислал Retry-After, пауза не короче него.
  401/402/403 с настоящей ошибкой про ключ банятся до конца сессии.

* 429 бывает двух разных видов, и лечатся они противоположно. «Rate Limit
  Error» — исчерпан лимит запросов, ключ уходит на паузу. А «Request too large
  for model ... on output tokens per minute (OTPM) ... reduce max_tokens» —
  это не лимит, а проверка потолка ответа: запрос отклонён целиком, потому что
  мы не отправили max_tokens и провайдер подставил свой дефолт выше лимита.
  Ключ при этом полностью жив и уводить его на паузу незачем: повтор с тем же
  телом упал бы точно так же. Такой отказ ничего не запоминает, обход модели
  прекращается, а следующего провайдера стоит попробовать — лимит относится к
  конкретной организации, у другого провайдера он может быть другим.

* 404 со структурированной ошибкой означает «у этого провайдера такой модели
  нет»: провайдер помечается недоступным для этой модели, и обход уходит к
  следующему провайдеру из списка модели. Без структурированной ошибки 404
  остаётся фатальным — непонятно, чья это была ошибка.

* 5xx не портит адреса. 502/503/504 и 522 говорят про посредника, их адрес и
  ломает; 500, 501 и прочие — это ответ самого провайдера, и виноват адрес не
  может. Раньше любой 5xx уводил адрес в bad_proxies до конца сессии, и одна
  неудача Gemini на auto1 выводила из строя auto1, auto2 и так далее, пока
  адреса не кончались; теперь такие ответы означают «эта модель у провайдера
  не отвечает» и уводят обход к следующему провайдеру, ничего не запоминая.

Сломанные ключи и адреса запоминаются в состоянии сессии, чтобы не перебирать
их повторно для каждой следующей модели. Адрес `direct` в сломанные не попадает:
обрыв через direct означает, что провайдер недоступен, а не адрес. Пометки
«провайдер не подходит» стираются на новом диалоге (SessionState.forget_models):
квота могла восстановиться, и новый чат должен снова попробовать модель.
"""

from __future__ import annotations

import logging
import threading
import time
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from datetime import datetime, timezone
from email.utils import parsedate_to_datetime
from enum import Enum
from importlib.util import find_spec
from typing import Self

import httpx

from .registry import (
    DEFAULT_API_STYLE,
    Binding,
    ConfigError,
    Model,
    Provider,
    Proxy,
    Registry,
    Route,
    load_registry,
)

log = logging.getLogger(__name__)

RATE_LIMIT_STATUS = 429
DEFAULT_COOLDOWN = 60.0
PROBE_MODEL = "__probe__"
ADDRESS_STATUSES = frozenset({502, 503, 504, 522})
QUOTA_WORDS = ("quota", "rate limit", "rate_limit", "лимит", "квот")
TOO_LARGE_WORDS = ("request too large", "reduce max_tokens", "expected output tokens")


class Failure(Enum):
    """Что считаем сломанным по итогам попытки."""

    NONE = "успех"
    KEY = "ключ"
    QUOTA = "пауза"
    PROXY = "адрес"
    PROVIDER = "провайдер"
    MODEL = "модель"
    FATAL = "запрос"
    TOO_LARGE = "запрос велик"


class TurnBudget:
    """Пределы одного хода: время и число попыток, общие для всех моделей.

    Создаётся на ход и переиспользуется между моделями цепочки, поэтому лимит
    считается на всю цепочку, а не на каждую модель отдельно.
    """

    __slots__ = ("deadline", "max_attempts", "spent", "started")

    def __init__(
        self,
        deadline: float | None = None,
        max_attempts: int | None = None,
    ) -> None:
        self.deadline = deadline
        self.max_attempts = max_attempts
        self.started = time.monotonic()
        self.spent = 0

    def exhausted(self) -> str:
        """Пустая строка — можно пробовать; иначе причина остановиться."""
        if self.max_attempts is not None and self.spent >= self.max_attempts:
            return f"исчерпан лимит попыток на ход ({self.max_attempts})"
        if self.deadline is not None:
            left = self.deadline - (time.monotonic() - self.started)
            if left <= 0:
                return f"ход не уложился в {self.deadline:.0f} с"
        return ""


class LLMError(Exception):
    """Общая ошибка слоя запросов: ответ так и не получен."""


class GatewayError(LLMError):
    """Не удалось выполнить запрос ни одной связкой.

    `attempts` — все попытки обхода до отказа: из них собирается сноска в чате,
    поэтому ошибка несёт не только текст, но и что именно пробовали.
    """

    attempts: tuple[Attempt, ...] = ()


@dataclass(slots=True)
class SessionState:
    """Что уже признано нерабочим за время сессии."""

    bad_keys: set[str] = field(default_factory=set)
    bad_proxies: set[str] = field(default_factory=set)
    cooldown_keys: dict[str, float] = field(default_factory=dict)
    unavailable: set[tuple[str, str]] = field(default_factory=set)

    def key_available(self, name: str) -> bool:
        if name in self.bad_keys:
            return False
        until = self.cooldown_keys.get(name)
        if until is None:
            return True
        if time.monotonic() >= until:
            del self.cooldown_keys[name]
            return True
        return False

    def cool_down(self, name: str, seconds: float) -> None:
        self.cooldown_keys[name] = time.monotonic() + seconds

    def provider_available(self, provider: str, model_name: str) -> bool:
        return (provider, model_name) not in self.unavailable

    def drop_provider(self, provider: str, model_name: str) -> None:
        self.unavailable.add((provider, model_name))

    def forget_models(self) -> None:
        """Снять пометки «провайдер не подходит» — начался новый диалог.

        Квота могла восстановиться, поэтому новый чат снова пробует все модели
        цепочки с начала. Ключи и адреса остаются помеченными: если адрес не
        отвечал, он не заработает и в следующем чате.
        """
        self.unavailable.clear()

    def waiting(self) -> float:
        """Сколько секунд ждать до ближайшего возврата ключа в ротацию."""
        if not self.cooldown_keys:
            return 0.0
        return max(0.0, min(self.cooldown_keys.values()) - time.monotonic())

    def retry_hint(self) -> str:
        """Человекочитаемая причина, почему попыток не было."""
        parts: list[str] = []
        if self.bad_keys:
            parts.append(f"ключи помечены: {', '.join(sorted(self.bad_keys))}")
        if self.bad_proxies:
            parts.append(f"адреса помечены: {', '.join(sorted(self.bad_proxies))}")
        wait = self.waiting()
        if wait > 0:
            parts.append(f"ключ на паузе ещё {wait:.0f} с")
        return "; ".join(parts)

    def usable_keys(self, names: Sequence[str]) -> list[str]:
        return [n for n in names if self.key_available(n)]

    def usable_proxies(self, proxies: Sequence[Proxy]) -> list[Proxy]:
        return [p for p in proxies if p.name not in self.bad_proxies]


@dataclass(frozen=True, slots=True)
class Attempt:
    """Результат одной попытки: связка (прокси, ключ) и чем она кончилась.

    `payload` — то, что попытка вернула: у сырого HTTP это httpx.Response,
    у вызова чата ChatResponse или LiveStream. Наполняется только при
    Failure.NONE, наружу отдаётся вместе с попыткой.
    """

    provider: str
    proxy: Proxy
    key_name: str
    failure: Failure
    status: int | None = None
    detail: str = ""
    model: str = ""
    retry_after: float | None = None
    payload: object = field(default=None, repr=False, compare=False)

    def __str__(self) -> str:
        what = f"{self.failure.value}"
        if self.status is not None:
            what += f" HTTP {self.status}"
        if self.detail:
            what += f" ({self.detail})"
        if self.retry_after is not None:
            what += f" [ждём {self.retry_after:.0f} с]"
        where = f"{self.provider}: {self.key_name}@{self.proxy.name}"
        return f"{self.model}: {where} → {what}" if self.model else f"{where} → {what}"


@dataclass(frozen=True, slots=True)
class Rotation:
    """Итог обхода: удачная попытка и всё, что было до неё.

    Обход возвращает только успех, но вызывающему нужно и то, что не
    сработало: по этим попыткам в чате печатается сноска «сначала был
    отказ», и по ним же видно, куда ушла ротация.
    """

    attempt: Attempt
    attempts: tuple[Attempt, ...] = ()

    @property
    def failed(self) -> tuple[Attempt, ...]:
        return tuple(a for a in self.attempts if a.failure is not Failure.NONE)

    @property
    def payload(self) -> object:
        return self.attempt.payload


Executor = Callable[[Route, Binding], Attempt]
"""Одна попытка: предъявить связку и вернуть Attempt.

Результат успешной попытки лежит в Attempt.payload, чем кончилась — в
Attempt.failure. Смысл такого разделения в том, что цикл обхода в Gateway
не разбирает ни тела запроса, ни формата ответа: это знает исполнитель.
"""


def provider_error(response: httpx.Response) -> dict | None:
    """Структурированная ошибка провайдера или None.

    None означает, что ответ пришёл не от API провайдера: Cloudflare, WAF,
    прокси-сервер и прочие посредники. Их ответы лечатся перебором адресов,
    а не ключей.
    """
    try:
        payload = response.json()
    except ValueError:
        return None
    if isinstance(payload, list) and len(payload) == 1:
        payload = payload[0]
    if not isinstance(payload, dict):
        return None
    error = payload.get("error")
    if isinstance(error, dict):
        return error
    if isinstance(error, str):
        return {"message": error}
    return None


def classify(response: httpx.Response) -> Failure:
    """Определить, что сломалось, по ответу.

    Проверка текста идёт до разбора статуса: «слишком большой запрос» Groq
    отдаёт кодом 429, тем же кодом, что и настоящий лимит запросов, но лечится
    уменьшением max_tokens в конфиге, а не паузой на ключ.
    """
    if is_too_large_error(response):
        return Failure.TOO_LARGE
    status = response.status_code
    structured = provider_error(response)
    if status in (401, 403):
        if status == 403 and is_quota_error(response):
            return Failure.QUOTA
        return Failure.KEY if structured else Failure.PROXY
    if status == 402:
        return Failure.KEY
    if status == RATE_LIMIT_STATUS:
        return Failure.QUOTA
    if status == 404:
        return Failure.PROVIDER if structured else Failure.FATAL
    if status in ADDRESS_STATUSES:
        return Failure.PROXY
    if status >= 500:
        return Failure.MODEL
    return Failure.FATAL


def is_too_large_error(response: httpx.Response) -> bool:
    """Провал проверки потолка ответа, а не исчерпание лимита.

    Текст такого отказа прямо просит уменьшить max_tokens. Ключ и адрес тут ни
    при чём, поэтому такой отказ не должен никого банить.
    """
    detail = error_detail(response).lower()
    return any(word in detail for word in TOO_LARGE_WORDS)


def is_quota_error(response: httpx.Response) -> bool:
    """Отказ из-за лимита у провайдера, а не из-за ключа.

    Google отдаёт 403 с текстом про квоту на вполне рабочем ключе: такой ключ
    надо просто подождать, а не выбрасывать до конца сессии.
    """
    detail = error_detail(response).lower()
    return any(word in detail for word in QUOTA_WORDS)


def retry_after(response: httpx.Response) -> float | None:
    """Сколько провайдер просит не дёргать его — Retry-After в секундах."""
    raw = response.headers.get("retry-after")
    if not raw:
        return None
    raw = raw.strip()
    try:
        return max(0.0, float(raw))
    except ValueError:
        pass
    try:
        when = parsedate_to_datetime(raw)
    except (TypeError, ValueError):
        return None
    if when.tzinfo is None:
        when = when.replace(tzinfo=timezone.utc)
    return max(0.0, (when - datetime.now(timezone.utc)).total_seconds())


class ApiAdapter:
    """Заголовки авторизации и разбор ответа со списком моделей."""

    def headers(self, key_value: str) -> dict[str, str]:
        return {
            "Authorization": f"Bearer {key_value}",
            "Content-Type": "application/json",
        }

    def parse_models(self, payload: object) -> list[str]:
        if not isinstance(payload, dict):
            raise ConfigError("Неожиданный формат ответа со списком моделей")
        data = payload.get("data")
        if not isinstance(data, list):
            raise ConfigError("Ожидалось поле 'data' со списком моделей")
        return [
            str(item["id"]) for item in data if isinstance(item, dict) and "id" in item
        ]


class GoogleNativeAdapter(ApiAdapter):
    """Нативный Gemini API: другой заголовок ключа и другой формат ответа."""

    def headers(self, key_value: str) -> dict[str, str]:
        return {
            "x-goog-api-key": key_value,
            "Content-Type": "application/json",
        }

    def parse_models(self, payload: object) -> list[str]:
        if not isinstance(payload, dict):
            raise ConfigError("Неожиданный формат ответа со списком моделей")
        data = payload.get("models")
        if not isinstance(data, list):
            raise ConfigError("Ожидалось поле 'models' со списком моделей")
        names = []
        for item in data:
            raw = item.get("name") if isinstance(item, dict) else None
            if isinstance(raw, str):
                names.append(raw.removeprefix("models/"))
        return names


API_STYLES: dict[str, type[ApiAdapter]] = {
    "openai": ApiAdapter,
    "google_native": GoogleNativeAdapter,
}


def adapter_for(provider: Provider) -> ApiAdapter:
    try:
        return API_STYLES[provider.api_style]()
    except KeyError:
        raise ConfigError(
            f"Провайдер {provider.name!r}: неизвестный api_style "
            f"{provider.api_style!r}. Известные: {', '.join(sorted(API_STYLES))}"
        ) from None


def _check_proxy_ready(proxy: Proxy) -> None:
    """Проверить, что адрес вообще можно использовать, до создания клиента.

    httpx для socks5 требует пакет socksio и без него падает внутри
    httpx.Client: ImportError в середине обхода ни о чём не говорит. Здесь
    сообщение объясняет, что доставить.
    """
    if proxy.is_socks and find_spec("socksio") is None:
        raise ConfigError(
            f"Прокси {proxy.name!r} ({proxy.httpx_proxy}) требует пакет socksio — "
            'он не установлен. Поставить: pip install "httpx[socks]"'
        )


class Gateway:
    """Ходит к моделям, перебирая провайдеров, ключи и адреса."""

    def __init__(
        self,
        registry: Registry | None = None,
        state: SessionState | None = None,
        read_timeout: float = 120.0,
        connect_timeout: float = 10.0,
        cooldown: float = DEFAULT_COOLDOWN,
        client: httpx.Client | None = None,
        turn_deadline: float | None = None,
        turn_max_attempts: int | None = None,
    ) -> None:
        self.registry = registry if registry is not None else load_registry()
        self.state = state if state is not None else SessionState()
        self.read_timeout = read_timeout
        self.connect_timeout = connect_timeout
        self.cooldown = cooldown
        self._external_client = client
        self._clients: dict[str, httpx.Client] = {}
        self._clients_lock = threading.Lock()
        self.turn_deadline = (
            self.registry.turn_deadline if turn_deadline is None else turn_deadline
        )
        self.turn_max_attempts = (
            self.registry.turn_max_attempts
            if turn_max_attempts is None
            else turn_max_attempts
        )

    def turn_budget(self) -> TurnBudget:
        """Новый бюджет хода с пределами из .env (или из аргументов Gateway)."""
        return TurnBudget(
            deadline=self.turn_deadline,
            max_attempts=self.turn_max_attempts,
        )

    def client(self, proxy: Proxy) -> httpx.Client:
        if self._external_client is not None:
            return self._external_client
        client = self._clients.get(proxy.name)
        if client is not None:
            return client
        _check_proxy_ready(proxy)
        with self._clients_lock:
            client = self._clients.get(proxy.name)
            if client is None:
                client = httpx.Client(
                    proxy=proxy.httpx_proxy,
                    timeout=httpx.Timeout(
                        self.read_timeout,
                        connect=self.connect_timeout,
                    ),
                    follow_redirects=True,
                )
                self._clients[proxy.name] = client
        return client

    def close(self) -> None:
        for client in self._clients.values():
            client.close()
        self._clients.clear()

    def __enter__(self) -> Self:
        return self

    def __exit__(self, *exc_info: object) -> None:
        self.close()

    def probe_route(self, provider_name: str) -> Route:
        """Синтетический маршрут по всем связкам провайдера — для служебных запросов."""
        provider = self.registry.provider(provider_name)
        bindings = self.registry.bindings_for(provider_name)
        if not bindings:
            raise ConfigError(
                f"Провайдер {provider.name!r}: "
                f"{self.registry.credential_problem(provider_name)}"
            )
        model = Model(name=PROBE_MODEL, providers=(provider_name,))
        return Route(model=model, provider=provider, bindings=bindings)

    def request(
        self,
        model_name: str,
        method: str,
        url: str,
        json: dict | None = None,
        params: dict | None = None,
        read_timeout: float | None = None,
    ) -> httpx.Response:
        """Выполнить запрос к модели, обходя её провайдеров по очереди."""
        return self.request_routes(
            self.registry.routes(model_name),
            method,
            url,
            json=json,
            params=params,
            read_timeout=read_timeout,
        )

    def request_routes(
        self,
        routes: Sequence[Route],
        method: str,
        url: str,
        json: dict | None = None,
        params: dict | None = None,
        read_timeout: float | None = None,
    ) -> httpx.Response:
        """Обойти маршруты сырым HTTP-запросом. Бросает GatewayError."""
        rotation = self.rotate(
            routes,
            lambda route, binding: self._http_attempt(
                route, binding, method, url, json, params, read_timeout
            ),
        )
        assert isinstance(rotation.payload, httpx.Response)
        return rotation.payload

    def rotate(
        self,
        routes: Sequence[Route],
        executor: Executor,
        budget: TurnBudget | None = None,
    ) -> Rotation:
        """Обойти маршруты, отдавая каждую попытку executor. Бросает GatewayError.

        Единственное место, где решается, что именно сломалось и куда идти
        дальше. Возвращает удачную попытку вместе со всеми предыдущими.
        Перебор останавливается на первом успехе, а также когда кончился
        TurnBudget: ход не должен висеть на семидесяти попытках.
        """
        limit = budget if budget is not None else self.turn_budget()
        attempts: list[Attempt] = []
        for route in routes:
            provider = route.provider.name
            model_name = route.model.name
            if not self.state.provider_available(provider, model_name):
                continue

            bindings = list(route.bindings)
            index = _live(bindings, 0, self.state)
            while index < len(bindings):
                reason = limit.exhausted()
                if reason:
                    raise self._fail(routes, attempts, reason)
                binding = bindings[index]
                limit.spent += 1
                attempt = executor(route, binding)
                attempts.append(attempt)
                log.debug(
                    "проба %d: %s%s",
                    limit.spent,
                    attempt,
                    " ← успех" if attempt.failure is Failure.NONE else "",
                )
                if attempt.failure is Failure.NONE:
                    return Rotation(attempt=attempt, attempts=tuple(attempts))

                self._remember(attempt)
                if attempt.failure is Failure.FATAL:
                    raise self._fail(
                        routes,
                        attempts,
                        f"{route.label}: запрос отклонён провайдером, перебор "
                        f"бессмыслен — {attempt}",
                    )
                if attempt.failure is Failure.PROVIDER:
                    self.state.drop_provider(provider, model_name)
                    break
                if attempt.failure is Failure.MODEL:
                    log.debug(
                        "%s: провайдер %s не отвечает по этой модели, адрес не трогаем",
                        model_name,
                        provider,
                    )
                    break
                if attempt.failure is Failure.TOO_LARGE:
                    log.debug(
                        "%s: провайдер %s отклонил потолок ответа, ключи целы",
                        model_name,
                        provider,
                    )
                    break
                if attempt.failure is Failure.PROXY:
                    index = _live(bindings, index + 1, self.state)
                else:
                    index = _live(
                        bindings, index + 1, self.state, skip_key=binding.key_name
                    )
        raise self._fail(routes, attempts, limit.exhausted())

    def chain_error(
        self,
        attempts: Sequence[Attempt],
        models: Sequence[str] = (),
        last: GatewayError | None = None,
    ) -> GatewayError:
        """Ошибка на весь ход, когда молчали все модели цепочки.

        Обрыв внутри модели показывает её попытки (`_summary`), а исчерпанная
        цепочка — все сразу и по моделям: иначе в сообщении видно только
        последнюю, и кажется, будто до Groq или локальной модели не дошли.
        Модели без попыток тоже перечисляются: обычно их нет потому, что адрес
        или ключ уже помечены первой моделью, и не сказать об этом — значит
        скрыть настоящую причину.
        """
        reason = ""
        if last is not None and last.args:
            head = str(last.args[0]).splitlines()[0]
            if head.startswith("обход прерван: "):
                reason = head.removeprefix("обход прерван: ").rstrip(".")

        grouped: dict[str, list[Attempt]] = {}
        for attempt in attempts:
            grouped.setdefault(attempt.model or "?", []).append(attempt)
        ordered = list(dict.fromkeys([*models, *grouped]))

        lines: list[str] = []
        if reason:
            lines.append(f"обход прерван: {reason}.")
        if ordered:
            lines.append("цепочка: ни одна модель не ответила.")
            for model_name in ordered:
                model_attempts = grouped.get(model_name, [])
                lines.append(f"  {model_name}:")
                if model_attempts:
                    lines.extend(f"    попытка: {a}" for a in model_attempts)
                elif reason:
                    lines.append("    попыток не было — обход прерван по лимиту хода")
                else:
                    lines.append("    попыток не было — ключи и адреса уже помечены")
        else:
            lines.append("цепочка: не удалось выполнить запрос, попыток не было.")
            hint = self.state.retry_hint()
            if hint:
                lines.append(f"  ключи, адреса или провайдеры уже помечены ({hint})")
        error = GatewayError("\n".join(lines))
        error.attempts = tuple(attempts)
        return error

    def _fail(
        self, routes: Sequence[Route], attempts: list[Attempt], reason: str
    ) -> GatewayError:
        """Ошибка обхода вместе со всеми попытками: чату нужны обе вещи."""
        error = GatewayError(self._summary(routes, attempts, reason))
        error.attempts = tuple(attempts)
        return error

    def _remember(self, attempt: Attempt) -> None:
        if attempt.failure is Failure.KEY:
            self.state.bad_keys.add(attempt.key_name)
        elif attempt.failure is Failure.QUOTA:
            pause = max(self.cooldown, attempt.retry_after or 0.0)
            self.state.cool_down(attempt.key_name, pause)
            log.info("ключ %s на паузе %.0f с", attempt.key_name, pause)
        elif attempt.failure is Failure.PROXY and not attempt.proxy.is_direct:
            self.state.bad_proxies.add(attempt.proxy.name)

    def _summary(
        self,
        routes: Sequence[Route],
        attempts: list[Attempt],
        budget_reason: str = "",
    ) -> str:
        lines: list[str] = []
        if budget_reason:
            lines.append(f"обход прерван: {budget_reason}.")
        if routes:
            head = routes[0].model.label if len(routes) == 1 else "цепочка"
            lines.append(f"{head}: не удалось выполнить запрос ни одной связкой.")
        else:
            lines.append("не удалось выполнить запрос: нет ни одного маршрута.")
        seen: list[str] = []
        for route in routes:
            if route.provider.name not in seen:
                seen.append(route.provider.name)
        if seen:
            lines.append("  провайдеры: " + ", ".join(seen))
        for route in routes:
            if not self.state.provider_available(route.provider.name, route.model.name):
                lines.append(
                    f"  {route.provider.name}: модели нет (помечен после 404), "
                    "за новым диалогом попробуем снова"
                )
        lines.extend(f"  попытка: {a}" for a in attempts)
        if any(a.failure is Failure.TOO_LARGE for a in attempts):
            lines.append(
                "  подсказка: провайдер отклонил потолок ответа. Уменьшите "
                "max_output_tokens у модели в models/models.toml."
            )
        if not attempts:
            hint = self.state.retry_hint()
            lines.append(
                "  попыток не было: ключи, адреса или провайдеры уже помечены "
                f"нерабочими в этой сессии ({hint})"
                if hint
                else "  попыток не было: ключи, адреса или провайдеры уже помечены "
                "нерабочими в этой сессии"
            )
        return "\n".join(lines)

    def _http_attempt(
        self,
        route: Route,
        binding: Binding,
        method: str,
        url: str,
        json: dict | None,
        params: dict | None,
        read_timeout: float | None,
    ) -> Attempt:
        adapter = adapter_for(route.provider)
        client = self.client(binding.proxy)
        timeout = (
            httpx.Timeout(read_timeout, connect=self.connect_timeout)
            if read_timeout is not None
            else None
        )
        headers = adapter.headers(binding.key_value)
        if route.provider.auth == "none":
            # У локального сервера ключа нет, а «Bearer » с пустым значением —
            # невалидный заголовок: httpx не отправляет такой запрос вообще.
            headers.pop("Authorization", None)
        log.debug(
            "%s %s через %s ключ %s",
            method,
            url,
            binding.proxy.name,
            binding.key_name,
        )
        try:
            response = client.request(
                method,
                url,
                json=json,
                params=params,
                headers=headers,
                timeout=timeout,
            )
        except httpx.TransportError as exc:
            return Attempt(
                model=route.model.name,
                provider=route.provider.name,
                proxy=binding.proxy,
                key_name=binding.key_name,
                failure=Failure.PROXY,
                detail=type(exc).__name__,
            )
        except httpx.HTTPError as exc:
            return Attempt(
                model=route.model.name,
                provider=route.provider.name,
                proxy=binding.proxy,
                key_name=binding.key_name,
                failure=Failure.FATAL,
                detail=f"{type(exc).__name__}: {exc}",
            )

        if response.is_success:
            return Attempt(
                model=route.model.name,
                provider=route.provider.name,
                proxy=binding.proxy,
                key_name=binding.key_name,
                failure=Failure.NONE,
                status=response.status_code,
                payload=response,
            )

        return Attempt(
            model=route.model.name,
            provider=route.provider.name,
            proxy=binding.proxy,
            key_name=binding.key_name,
            failure=classify(response),
            status=response.status_code,
            detail=error_detail(response),
            retry_after=retry_after(response),
        )

    def list_models(
        self, provider_name: str, read_timeout: float | None = None
    ) -> list[str]:
        """Получить у провайдера полный список доступных моделей."""
        route = self.probe_route(provider_name)
        provider = route.provider
        response = self.request_routes(
            [route],
            "GET",
            provider.models_url,
            read_timeout=read_timeout,
        )
        try:
            payload = response.json()
        except ValueError as exc:
            raise GatewayError(
                f"{provider.name}: ответ со списком моделей не является JSON — {exc}"
            ) from None
        names = adapter_for(provider).parse_models(payload)
        return [provider.strip_id_prefix(n) for n in names]

    def chat(self, model_name: str, messages: list[dict], **params: object) -> dict:
        """Устарело: собирало тело вручную и не знало про перебор связок.

        Живой вызов чата — это LLMClient из llm_client, он ходит через
        openai-совместимый слой и через тот же цикл обхода Gateway.rotate.
        """
        raise NotImplementedError(
            "Gateway.chat удалён, используй LLMClient.complete()/stream() из llm_client"
        )


def _live(
    bindings: list[Binding],
    start: int,
    state: SessionState,
    skip_key: str | None = None,
) -> int:
    """Индекс следующей годной связки: ключ жив, адрес не помечен, ключ не пропускаемый."""
    index = start
    while index < len(bindings):
        binding = bindings[index]
        if binding.key_name == skip_key:
            index += 1
            continue
        if not state.key_available(binding.key_name):
            index += 1
            continue
        if binding.proxy.name in state.bad_proxies:
            index += 1
            continue
        return index
    return len(bindings)


def _error_detail(error: dict | None) -> str:
    if not error:
        return ""
    for field_name in ("message", "type", "status"):
        value = error.get(field_name)
        if isinstance(value, str) and value.strip():
            return value.strip()[:300]
    return ""


def _short_text(response: httpx.Response) -> str:
    return response.text.strip().replace("\n", " ")[:200]


def error_detail(response: httpx.Response) -> str:
    """Человекочитаемая причина отказа из тела ответа.

    Единая точка для обоих путей: и сырого HTTP здесь, и openai-совместимого
    слоя в llm_transport, где у исключения SDK лежит тот же httpx.Response.
    """
    return _error_detail(provider_error(response)) or _short_text(response)


__all__ = [
    "API_STYLES",
    "DEFAULT_API_STYLE",
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
]
