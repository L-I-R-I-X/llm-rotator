"""Тесты ModelRunner: обход связок, параметры запроса, поток, жизненный цикл.

Сети нет: HTTPS отвечает MockTransport со сценарием, поэтому видно и тело
каждого запроса, и то, через какой ключ и адрес он ушёл.
"""

from __future__ import annotations

import json
import unittest
from importlib.util import find_spec

import httpx

from llm_rotator import (
    LOCAL_API_KEY,
    Chain,
    ConfigError,
    Failure,
    Gateway,
    GatewayError,
    Key,
    Model,
    ModelRunner,
    Provider,
    Proxy,
    Registry,
    SessionState,
    StreamBroken,
    classify,
    failover_notes,
)


def completion(
    text: str = "Привет",
    *,
    finish: str = "stop",
    usage: dict | None = None,
    tool_calls: list[dict] | None = None,
    model: str = "m",
) -> dict:
    """Тело обычного ответа в том виде, в каком его ждёт openai."""
    message: dict = {"role": "assistant", "content": text}
    if tool_calls is not None:
        message["tool_calls"] = tool_calls
    return {
        "id": "chatcmpl-1",
        "object": "chat.completion",
        "created": 1,
        "model": model,
        "choices": [{"index": 0, "message": message, "finish_reason": finish}],
        "usage": usage
        or {"prompt_tokens": 11, "completion_tokens": 7, "total_tokens": 18},
    }


def chunk(delta: dict, finish: str | None = None, usage: dict | None = None) -> dict:
    body: dict = {
        "id": "chatcmpl-1",
        "object": "chat.completion.chunk",
        "created": 1,
        "model": "m",
        "choices": [{"index": 0, "delta": delta, "finish_reason": finish}],
    }
    if usage is not None:
        body["usage"] = usage
    return body


def sse(*chunks: dict, done: bool = True) -> str:
    text = "".join("data: " + json.dumps(c) + "\n\n" for c in chunks)
    return text + ("data: [DONE]\n\n" if done else "")


def error(status: int, message: str = "отказ") -> httpx.Response:
    return httpx.Response(
        status, json={"error": {"message": message, "type": "invalid_request_error"}}
    )


class BrokenStream(httpx.SyncByteStream):
    """Поток, который рвётся после первого чанка."""

    def __init__(self, first: dict) -> None:
        self._first = first

    def __iter__(self):
        yield ("data: " + json.dumps(self._first) + "\n\n").encode()
        raise httpx.ReadError("connection reset by peer")


class StubGateway(Gateway):
    """Шлюз, вместо сети отдающий заранее расписанные ответы.

    Клиенты на каждый адрес — как у настоящего Gateway: только так видно,
    что запрос ушёл тем адресом, которым его назвали, а не первым попавшимся.
    """

    def __init__(
        self,
        registry: Registry,
        script: list | None = None,
        state: SessionState | None = None,
    ) -> None:
        self.script = list(script or [])
        self.calls: list[str] = []
        self.requests: list[dict] = []
        self._names = {value: name for name, value in registry.env_keys.items()}
        super().__init__(registry=registry, state=state)

    def client(self, proxy: Proxy) -> httpx.Client:
        known = self._clients.get(proxy.name)
        if known is not None:
            return known

        def handler(request: httpx.Request) -> httpx.Response:
            sent = request.headers.get("Authorization", "").removeprefix("Bearer ")
            key = self._names.get(sent, sent or "(без ключа)")
            self.calls.append(f"{key}@{proxy.name}")
            self.requests.append(
                {
                    "url": str(request.url),
                    "key_name": key,
                    "proxy_name": proxy.name,
                    "headers": dict(request.headers),
                    "body": json.loads(request.content) if request.content else {},
                }
            )
            reply = (
                self.script.pop(0)
                if self.script
                else error(500, "скрипт ответов кончился")
            )
            if isinstance(reply, httpx.Response):
                return reply
            if isinstance(reply, str):
                return httpx.Response(
                    200, text=reply, headers={"content-type": "text/event-stream"}
                )
            return httpx.Response(200, json=reply)

        known = httpx.Client(transport=httpx.MockTransport(handler))
        self._clients[proxy.name] = known
        return known


def build_registry(
    *,
    keys: tuple[tuple[str, str, tuple[str, ...]], ...] = (
        ("K1", "p", ("p1", "p2")),
        ("K2", "p", ("p3",)),
    ),
    models: tuple[Model, ...] = (Model("m", ("p",)),),
    providers: tuple[Provider, ...] = (
        Provider("p", "openai", "https://api.test/v1", "https://api.test/v1/models"),
    ),
    env_keys: dict[str, str] | None = None,
    chain: str = "main",
    role: str = "persona",
) -> Registry:
    """Реестр с одной цепочкой из всех моделей и ролью persona на неё."""
    return Registry(
        providers={p.name: p for p in providers},
        proxies={f"p{i}": Proxy(f"p{i}", f"http://p{i}:3128") for i in range(1, 5)},
        keys={name: Key(name, provider, proxies) for name, provider, proxies in keys},
        models={m.name: m for m in models},
        env_keys=env_keys
        if env_keys is not None
        else {k[0]: f"secret-{k[0]}" for k in keys},
        chains={chain: Chain(name=chain, models=tuple(models))},
        roles={role: chain},
    )


class RealGatewayProxyTests(unittest.TestCase):
    """Настоящий Gateway: адрес прокси должен дойти до httpx как URL со схемой."""

    def gateway(self, *proxies: Proxy) -> Gateway:
        reg = build_registry(
            keys=(("K1", "p", tuple(p.name for p in proxies)),),
            providers=(
                Provider(
                    "p",
                    "openai",
                    "https://api.test/v1",
                    "https://api.test/v1/models",
                ),
            ),
        )
        registry = Registry(
            providers=reg.providers,
            proxies=reg.proxies | {p.name: p for p in proxies},
            keys=reg.keys,
            models=reg.models,
            env_keys=reg.env_keys,
        )
        return Gateway(registry=registry)

    def test_address_without_scheme_is_accepted(self):
        gateway = self.gateway(Proxy("p1", "127.0.0.1:1091"))
        client = gateway.client(gateway.registry.proxy("p1"))
        self.assertIsNotNone(client)
        client.close()

    def test_socks5_client_is_built_when_socksio_is_installed(self):
        gateway = self.gateway(Proxy("p1", "127.0.0.1:1080", proxy_type="socks5"))
        proxy = gateway.registry.proxy("p1")
        if find_spec("socksio") is None:
            with self.assertRaises(ConfigError) as ctx:
                gateway.client(proxy)
            self.assertIn("socksio", str(ctx.exception))
            return
        gateway.client(proxy).close()

    def test_socks5_url_reaches_httpx(self):
        gateway = self.gateway(Proxy("p1", "127.0.0.1:1080", proxy_type="socks5"))
        proxy = gateway.registry.proxy("p1")
        self.assertEqual(proxy.httpx_proxy, "socks5://127.0.0.1:1080")


class RunnerCase(unittest.TestCase):
    script: list = ()

    def make(
        self,
        script: list | None = None,
        *,
        registry: Registry | None = None,
        state: SessionState | None = None,
    ) -> tuple[Registry, StubGateway, ModelRunner]:
        reg = registry if registry is not None else build_registry()
        gateway = StubGateway(
            reg, list(self.script) if script is None else script, state=state
        )
        return reg, gateway, ModelRunner(registry=reg, gateway=gateway)


class CompleteTests(RunnerCase):
    def test_text_and_origin_come_back(self):
        _, gateway, runner = self.make(
            [httpx.Response(200, json=completion("Привет, я Юля"))]
        )
        response = runner.complete("m", [{"role": "user", "content": "привет"}])
        self.assertEqual(response.text, "Привет, я Юля")
        self.assertEqual(response.origin.line(), "m · p · K1 · p1")
        self.assertEqual(response.origin.label, "m @ p")
        self.assertEqual(gateway.calls, ["K1@p1"])

    def test_origin_never_shows_key_value(self):
        _, gateway, runner = self.make([httpx.Response(200, json=completion())])
        response = runner.complete("m", [{"role": "user", "content": "привет"}])
        for text in (response.origin.line(), str(response.origin), response.text):
            self.assertNotIn("secret-K1", text)
        for note in failover_notes(response.attempts):
            self.assertNotIn("secret-K1", note)
        self.assertNotIn(
            "secret-K1",
            json.dumps(gateway.requests[0]["headers"].get("Authorization", "")),
        )

    def test_usage_and_finish_reason(self):
        usage = {
            "prompt_tokens": 100,
            "completion_tokens": 20,
            "total_tokens": 120,
            "prompt_tokens_details": {"cached_tokens": 90},
        }
        _, _, runner = self.make([httpx.Response(200, json=completion(usage=usage))])
        response = runner.complete("m", [{"role": "user", "content": "привет"}])
        self.assertEqual(response.usage.prompt_tokens, 100)
        self.assertEqual(response.usage.completion_tokens, 20)
        self.assertEqual(response.usage.total_tokens, 120)
        self.assertEqual(response.usage.cached_tokens, 90)
        self.assertEqual(response.finish_reason, "stop")
        self.assertEqual(str(response.usage), "100+20=120")

    def test_no_failover_notes_when_first_try_works(self):
        _, _, runner = self.make([httpx.Response(200, json=completion())])
        response = runner.complete("m", [{"role": "user", "content": "привет"}])
        self.assertEqual(response.attempts, ())
        self.assertEqual(failover_notes(response.attempts), ())

    def test_empty_content_is_empty_text(self):
        body = completion(None)
        _, _, runner = self.make([httpx.Response(200, json=body)])
        self.assertEqual(
            runner.complete("m", [{"role": "user", "content": "x"}]).text, ""
        )

    def test_reasoning_is_captured(self):
        body = completion()
        body["choices"][0]["message"]["reasoning"] = "  Я подумала  "
        _, _, runner = self.make([httpx.Response(200, json=body)])
        self.assertEqual(
            runner.complete("m", [{"role": "user", "content": "x"}]).reasoning,
            "Я подумала",
        )


class RotationTests(RunnerCase):
    def test_binding_order_is_key_then_proxy(self):
        reg, _, runner = self.make([httpx.Response(200, json=completion())])
        self.assertEqual(
            [str(b) for b in reg.routes("m")[0].bindings], ["K1@p1", "K1@p2", "K2@p3"]
        )
        runner.complete("m", [{"role": "user", "content": "x"}])

    def test_401_moves_to_next_key(self):
        _, gateway, runner = self.make(
            [error(401, "bad key"), httpx.Response(200, json=completion())]
        )
        response = runner.complete("m", [{"role": "user", "content": "x"}])
        self.assertEqual(gateway.calls, ["K1@p1", "K2@p3"])
        self.assertEqual(response.origin.key_name, "K2")

    def test_401_does_not_retry_the_same_key_on_another_proxy(self):
        _, gateway, runner = self.make(
            [error(401), httpx.Response(200, json=completion())]
        )
        runner.complete("m", [{"role": "user", "content": "x"}])
        self.assertNotIn("K1@p2", gateway.calls)

    def test_429_moves_to_next_key_and_survives_cooldown(self):
        state = SessionState()
        _, gateway, runner = self.make(
            [error(429, "rate")] + [httpx.Response(200, json=completion())] * 3,
            state=state,
        )
        first = runner.complete("m", [{"role": "user", "content": "x"}])
        self.assertEqual(first.origin.key_name, "K2")
        self.assertEqual(state.usable_keys(["K1", "K2"]), ["K2"])
        for _ in range(2):
            runner.complete("m", [{"role": "user", "content": "x"}])
        self.assertEqual(gateway.calls, ["K1@p1", "K2@p3", "K2@p3", "K2@p3"])

    def test_404_moves_to_next_provider(self):
        reg = build_registry(
            keys=(("K1", "p", ("p1",)), ("K2", "p2", ("p2",))),
            models=(Model("m", ("p", "p2")),),
            providers=(
                Provider(
                    "p", "openai", "https://api.test/v1", "https://api.test/v1/models"
                ),
                Provider(
                    "p2",
                    "openai",
                    "https://other.test/v1",
                    "https://other.test/v1/models",
                ),
            ),
        )
        _, gateway, runner = self.make(
            [error(404, "нет модели"), httpx.Response(200, json=completion())],
            registry=reg,
        )
        response = runner.complete("m", [{"role": "user", "content": "x"}])
        self.assertEqual(response.origin.provider, "p2")
        self.assertEqual(gateway.calls, ["K1@p1", "K2@p2"])

    def test_502_moves_to_next_proxy_keeping_key(self):
        """502 говорит про посредника: адрес плохой, ключ тот же."""
        _, gateway, runner = self.make(
            [error(502, "плохой шлюз"), httpx.Response(200, json=completion())]
        )
        response = runner.complete("m", [{"role": "user", "content": "x"}])
        self.assertEqual(gateway.calls, ["K1@p1", "K1@p2"])
        self.assertEqual(response.origin.proxy_name, "p2")

    def test_500_means_model_is_down_and_leaves_address_alone(self):
        """500 — это ответ самого провайдера, адрес тут ни при чём."""
        reg = build_registry(
            keys=(("K1", "p", ("p1", "p2")), ("K2", "p2", ("p2",))),
            models=(Model("m", ("p", "p2")),),
            providers=(
                Provider(
                    "p", "openai", "https://api.test/v1", "https://api.test/v1/models"
                ),
                Provider(
                    "p2",
                    "openai",
                    "https://other.test/v1",
                    "https://other.test/v1/models",
                ),
            ),
        )
        _, gateway, runner = self.make(
            [error(500, "упало"), httpx.Response(200, json=completion())], registry=reg
        )
        response = runner.complete("m", [{"role": "user", "content": "x"}])
        self.assertEqual(response.origin.provider, "p2")
        # ушли к другому провайдеру, а не к другому адресу того же ключа
        self.assertEqual(gateway.calls, ["K1@p1", "K2@p2"])
        self.assertEqual(gateway.state.bad_proxies, set())
        self.assertEqual(gateway.state.unavailable, set())

    def test_400_stops_rotation_immediately(self):
        _, gateway, runner = self.make(
            [error(400, "плохой параметр"), httpx.Response(200, json=completion())]
        )
        with self.assertRaises(GatewayError) as ctx:
            runner.complete("m", [{"role": "user", "content": "x"}])
        self.assertEqual(gateway.calls, ["K1@p1"])
        self.assertIn("плохой параметр", str(ctx.exception))

    def test_422_stops_rotation_immediately(self):
        _, gateway, runner = self.make(
            [error(422, "невалидно"), httpx.Response(200, json=completion())]
        )
        with self.assertRaises(GatewayError):
            runner.complete("m", [{"role": "user", "content": "x"}])
        self.assertEqual(gateway.calls, ["K1@p1"])

    def test_everything_dead_raises_with_attempts_listed(self):
        _, gateway, runner = self.make(
            [error(502, "упало"), error(502, "упало"), error(502, "упало")]
        )
        with self.assertRaises(GatewayError) as ctx:
            runner.complete("m", [{"role": "user", "content": "x"}])
        self.assertEqual(gateway.calls, ["K1@p1", "K1@p2", "K2@p3"])
        self.assertIn("упало", str(ctx.exception))

    def test_sdk_does_not_retry_by_itself(self):
        reg = build_registry(keys=(("K1", "p", ("p1", "p2")),))
        _, gateway, runner = self.make([error(401, "сломанный ключ")], registry=reg)
        with self.assertRaises(GatewayError):
            runner.complete("m", [{"role": "user", "content": "x"}])
        self.assertEqual(gateway.calls, ["K1@p1"])

    def test_connection_error_is_treated_as_broken_proxy(self):
        def handler(request):
            raise httpx.ConnectError("соединение отказало")

        reg = build_registry(keys=(("K1", "p", ("p1", "p2")),))
        gateway = StubGateway(reg)
        gateway.client = lambda proxy: httpx.Client(  # type: ignore[method-assign]
            transport=httpx.MockTransport(handler)
        )
        runner = ModelRunner(registry=reg, gateway=gateway)
        with self.assertRaises(GatewayError) as ctx:
            runner.complete("m", [{"role": "user", "content": "x"}])
        self.assertIn("p1", str(ctx.exception))
        self.assertIn("p2", str(ctx.exception))

    def test_failover_notes_describe_only_failed_attempts(self):
        _, _, runner = self.make(
            [error(502, "упало"), httpx.Response(200, json=completion())]
        )
        response = runner.complete("m", [{"role": "user", "content": "x"}])
        notes = failover_notes(response.attempts)
        self.assertEqual(len(notes), 1)
        self.assertIn("K1@p1", notes[0])
        self.assertIn("упало", notes[0])

    def test_note_names_the_model(self):
        _, _, runner = self.make(
            [error(502, "упало"), httpx.Response(200, json=completion())]
        )
        response = runner.complete("m", [{"role": "user", "content": "x"}])
        self.assertIn("m:", failover_notes(response.attempts)[0])


class OutputCapTests(RunnerCase):
    """Потолок ответа: конфиг, явный параметр и отказ провайдера."""

    def reg(self, cap: int | None, **kwargs) -> Registry:
        return build_registry(
            models=(Model("m", ("p",), max_output_tokens=cap),),
            **kwargs,
        )

    def two_providers(self, cap: int | None) -> Registry:
        return build_registry(
            keys=(("K1", "p", ("p1", "p2")), ("K2", "p2", ("p3",))),
            models=(Model("m", ("p", "p2"), max_output_tokens=cap),),
            providers=(
                Provider(
                    "p", "openai", "https://api.test/v1", "https://api.test/v1/models"
                ),
                Provider(
                    "p2",
                    "openai",
                    "https://other.test/v1",
                    "https://other.test/v1/models",
                ),
            ),
        )

    def test_cap_from_config_reaches_request_body(self):
        """Groq отклоняет запрос без max_tokens: подставим потолок из TOML."""
        _, gateway, runner = self.make(
            [httpx.Response(200, json=completion())], registry=self.reg(999)
        )
        runner.complete("m", [{"role": "user", "content": "x"}])
        self.assertEqual(gateway.requests[0]["body"]["max_tokens"], 999)

    def test_no_cap_means_no_max_tokens(self):
        """Модель без потолка в конфиге — запрос ровно как раньше."""
        _, gateway, runner = self.make(
            [httpx.Response(200, json=completion())], registry=self.reg(None)
        )
        runner.complete("m", [{"role": "user", "content": "x"}])
        self.assertNotIn("max_tokens", gateway.requests[0]["body"])

    def test_explicit_max_tokens_overrides_config(self):
        _, gateway, runner = self.make(
            [httpx.Response(200, json=completion())], registry=self.reg(999)
        )
        runner.complete("m", [{"role": "user", "content": "x"}], max_tokens=50)
        self.assertEqual(gateway.requests[0]["body"]["max_tokens"], 50)

    def oversized(self) -> httpx.Response:
        return error(
            429,
            "Request too large for model qwen/qwen3.8-27b on output tokens per "
            "minute (OTPM): Limit 1000, Requested 1653. Please reduce max_tokens.",
        )

    def test_oversized_429_is_not_a_rate_limit(self):
        self.assertIs(classify(self.oversized()), Failure.TOO_LARGE)
        self.assertIs(classify(error(429, "Rate Limit Error")), Failure.QUOTA)

    def test_oversized_429_keeps_the_key_alive(self):
        """Повтор с тем же телом упал бы так же — ключ на паузу не уходит."""
        state = SessionState()
        _, gateway, runner = self.make(
            [self.oversized()], registry=self.reg(999), state=state
        )
        with self.assertRaises(GatewayError):
            runner.complete("m", [{"role": "user", "content": "x"}])
        self.assertEqual(state.usable_keys(["K1", "K2"]), ["K1", "K2"])
        self.assertEqual(gateway.state.bad_keys, set())
        self.assertEqual(state.cooldown_keys, {})

    def test_oversized_429_does_not_try_another_key_of_same_provider(self):
        _, gateway, runner = self.make(
            [self.oversized(), httpx.Response(200, json=completion())],
            registry=self.two_providers(999),
        )
        response = runner.complete("m", [{"role": "user", "content": "x"}])
        self.assertEqual(gateway.calls, ["K1@p1", "K2@p3"])
        self.assertEqual(response.origin.provider, "p2")

    def test_oversized_alone_reports_the_hint(self):
        _, _, runner = self.make([self.oversized()], registry=self.reg(999))
        with self.assertRaises(GatewayError) as ctx:
            runner.complete("m", [{"role": "user", "content": "x"}])
        self.assertIn("max_output_tokens", str(ctx.exception))
        self.assertIn("models.toml", str(ctx.exception))


class RequestTests(RunnerCase):
    def test_params_reach_request_body(self):
        tools = [{"type": "function", "function": {"name": "поиск", "parameters": {}}}]
        _, gateway, runner = self.make([httpx.Response(200, json=completion())])
        runner.complete(
            "m",
            [{"role": "user", "content": "x"}],
            temperature=0.2,
            max_tokens=512,
            tools=tools,
            extra_body={"top_k": 40},
        )
        body = gateway.requests[0]["body"]
        self.assertEqual(body["model"], "m")
        self.assertEqual(body["temperature"], 0.2)
        self.assertEqual(body["max_tokens"], 512)
        self.assertEqual(body["tools"], tools)
        self.assertEqual(body["top_k"], 40)
        self.assertIs(body["stream"], False)

    def test_read_timeout_reaches_the_sdk_call(self):
        _, _, runner = self.make([httpx.Response(200, json=completion())])
        response = runner.complete(
            "m", [{"role": "user", "content": "x"}], read_timeout=12.5
        )
        self.assertEqual(response.text, "Привет")

    def test_chat_url_is_built_from_base_url(self):
        _, gateway, runner = self.make([httpx.Response(200, json=completion())])
        runner.complete("m", [{"role": "user", "content": "x"}])
        self.assertEqual(
            gateway.requests[0]["url"], "https://api.test/v1/chat/completions"
        )

    def test_model_id_prefix_is_stripped_per_route(self):
        reg = build_registry(
            keys=(("K1", "google", ("p1",)),),
            models=(Model("models/gemini-3.8-flash", ("google",)),),
            providers=(
                Provider(
                    "google",
                    "openai",
                    "https://google.test/v1",
                    "https://google.test/v1/models",
                    model_id_prefix="models/",
                ),
            ),
            env_keys={"K1": "secret-K1"},
        )
        _, gateway, runner = self.make(
            [httpx.Response(200, json=completion(model="gemini-3.8-flash"))],
            registry=reg,
        )
        response = runner.complete(
            "models/gemini-3.8-flash", [{"role": "user", "content": "x"}]
        )
        self.assertEqual(gateway.requests[0]["body"]["model"], "gemini-3.8-flash")
        self.assertEqual(response.origin.model, "models/gemini-3.8-flash")

    def test_extra_headers_are_sent(self):
        reg = build_registry(
            keys=(("K1", "p", ("p1",)),),
            providers=(
                Provider(
                    "p",
                    "openai",
                    "https://api.test/v1",
                    "https://api.test/v1/models",
                    extra_headers=(
                        ("X-Title", "yulya-bot"),
                        ("HTTP-Referer", "https://example.org"),
                    ),
                ),
            ),
        )
        _, gateway, runner = self.make(
            [httpx.Response(200, json=completion())], registry=reg
        )
        runner.complete("m", [{"role": "user", "content": "x"}])
        headers = gateway.requests[0]["headers"]
        self.assertEqual(headers["x-title"], "yulya-bot")
        self.assertEqual(headers["http-referer"], "https://example.org")

    def test_bearer_key_is_sent_in_authorization(self):
        _, gateway, runner = self.make([httpx.Response(200, json=completion())])
        runner.complete("m", [{"role": "user", "content": "x"}])
        self.assertEqual(
            gateway.requests[0]["headers"]["authorization"], "Bearer secret-K1"
        )

    def test_stream_flag_in_params_is_rejected_before_request(self):
        _, gateway, runner = self.make([httpx.Response(200, json=completion())])
        with self.assertRaises(GatewayError) as ctx:
            runner.complete("m", [{"role": "user", "content": "x"}], stream=True)
        self.assertIn("stream()", str(ctx.exception))
        self.assertEqual(gateway.calls, [])

    def test_unknown_model_fails_before_any_request(self):
        _, gateway, runner = self.make([httpx.Response(200, json=completion())])
        with self.assertRaises(ConfigError):
            runner.complete("нет-такой", [{"role": "user", "content": "x"}])
        self.assertEqual(gateway.calls, [])

    def test_unknown_api_style_is_reported_with_known_styles(self):
        reg = build_registry(
            keys=(("K1", "p", ("p1",)),),
            providers=(
                Provider(
                    "p",
                    "gemini_native",
                    "https://api.test/v1",
                    "https://api.test/v1/models",
                ),
            ),
        )
        _, gateway, runner = self.make([], registry=reg)
        with self.assertRaises(ConfigError) as ctx:
            runner.complete("m", [{"role": "user", "content": "x"}])
        self.assertIn("gemini_native", str(ctx.exception))
        self.assertIn("'openai'", str(ctx.exception))
        self.assertEqual(gateway.calls, [])


class AuthNoneTests(RunnerCase):
    def reg(self) -> Registry:
        return build_registry(
            keys=(("OLLAMA_LOCAL", "local", ("p1",)),),
            models=(Model("tiny", ("local",)),),
            providers=(
                Provider(
                    "local",
                    "openai",
                    "http://127.0.0.1:11434/v1",
                    "http://127.0.0.1:11434/v1/models",
                    auth="none",
                ),
            ),
            env_keys={},
        )

    def test_local_provider_needs_no_key_and_gets_placeholder(self):
        _, gateway, runner = self.make(
            [httpx.Response(200, json=completion(model="tiny"))], registry=self.reg()
        )
        response = runner.complete("tiny", [{"role": "user", "content": "x"}])
        request = gateway.requests[0]
        self.assertEqual(request["url"], "http://127.0.0.1:11434/v1/chat/completions")
        self.assertEqual(request["headers"]["authorization"], f"Bearer {LOCAL_API_KEY}")
        self.assertEqual(response.origin.line(), "tiny · local · OLLAMA_LOCAL · p1")

    def test_local_provider_is_usable_with_empty_env(self):
        reg = self.reg()
        with ModelRunner(registry=reg, gateway=StubGateway(reg)):
            pass
        self.assertEqual(reg.warnings, [])
        self.assertEqual(
            [str(b) for b in reg.route("tiny", "local").bindings], ["OLLAMA_LOCAL@p1"]
        )


class StreamTests(RunnerCase):
    def test_origin_is_known_before_first_read(self):
        _, gateway, runner = self.make(
            [sse(chunk({"role": "assistant", "content": "Пр"}))]
        )
        stream = runner.stream("m", [{"role": "user", "content": "x"}])
        self.assertEqual(stream.origin.line(), "m · p · K1 · p1")
        self.assertEqual(gateway.calls, ["K1@p1"])
        stream.close()

    def test_text_and_usage_are_collected(self):
        body = sse(
            chunk({"role": "assistant", "content": "Пр"}),
            chunk({"content": "ив"}),
            chunk(
                {},
                finish="stop",
                usage={"prompt_tokens": 7, "completion_tokens": 3, "total_tokens": 10},
            ),
        )
        _, _, runner = self.make([body])
        stream = runner.stream("m", [{"role": "user", "content": "x"}])
        self.assertEqual([item.delta for item in stream], ["Пр", "ив", ""])
        response = stream.collect()
        self.assertEqual(response.text, "Прив")
        self.assertEqual(response.usage.total_tokens, 10)
        self.assertEqual(response.finish_reason, "stop")
        self.assertEqual(response.origin.line(), "m · p · K1 · p1")

    def test_deltas_arrive_in_order(self):
        body = sse(
            chunk({"content": "раз"}),
            chunk({"content": ", "}),
            chunk({"content": "два"}),
        )
        _, _, runner = self.make([body])
        stream = runner.stream("m", [{"role": "user", "content": "x"}])
        deltas = [item.delta for item in stream]
        self.assertEqual(deltas, ["раз", ", ", "два"])
        self.assertEqual(stream.collect().text, "раз, два")

    def test_binding_is_switched_before_the_first_chunk(self):
        body = sse(chunk({"content": "ок"}))
        _, gateway, runner = self.make([error(503, "подождите"), body])
        stream = runner.stream("m", [{"role": "user", "content": "x"}])
        self.assertEqual(gateway.calls, ["K1@p1", "K1@p2"])
        self.assertEqual(stream.origin.key_name, "K1")
        self.assertEqual(stream.origin.proxy_name, "p2")
        self.assertEqual([item.delta for item in stream], ["ок"])
        self.assertEqual(stream.collect().text, "ок")
        self.assertEqual(len(failover_notes(stream.attempts)), 1)

    def test_tool_call_deltas_are_merged_by_index(self):
        body = sse(
            chunk(
                {
                    "role": "assistant",
                    "tool_calls": [
                        {
                            "index": 1,
                            "id": "call_2",
                            "type": "function",
                            "function": {"name": "второй", "arguments": '{"n":'},
                        }
                    ],
                }
            ),
            chunk(
                {
                    "tool_calls": [
                        {
                            "index": 0,
                            "id": "call_1",
                            "type": "function",
                            "function": {"name": "первый", "arguments": '{"q":'},
                        }
                    ]
                }
            ),
            chunk({"tool_calls": [{"index": 1, "function": {"arguments": " 2}"}}]}),
            chunk(
                {"tool_calls": [{"index": 0, "function": {"arguments": '"слова"}'}}]},
                finish="tool_calls",
            ),
        )
        _, _, runner = self.make([body])
        stream = runner.stream("m", [{"role": "user", "content": "x"}])
        list(stream)
        response = stream.collect()
        self.assertEqual([c.id for c in response.tool_calls], ["call_1", "call_2"])
        self.assertEqual(dict(response.tool_calls[0].arguments), {"q": "слова"})
        self.assertEqual(dict(response.tool_calls[1].arguments), {"n": 2})
        self.assertEqual(response.finish_reason, "tool_calls")

    def test_break_after_start_keeps_partial_text_and_does_not_retry(self):
        reply = httpx.Response(
            200,
            stream=BrokenStream(chunk({"content": "Начал"})),
            headers={"content-type": "text/event-stream"},
        )
        _, gateway, runner = self.make([reply])
        with (
            self.assertRaises(StreamBroken) as ctx,
            runner.stream("m", [{"role": "user", "content": "x"}]) as stream,
        ):
            list(stream)
        self.assertEqual(ctx.exception.partial_text, "Начал")
        self.assertIn("поток оборвался", str(ctx.exception))
        self.assertEqual(gateway.calls, ["K1@p1"])

    def test_empty_stream_is_fatal_without_retry(self):
        _, gateway, runner = self.make([""])
        with self.assertRaises(GatewayError) as ctx:
            runner.stream("m", [{"role": "user", "content": "x"}])
        self.assertIn("ни одного чанка", str(ctx.exception))
        self.assertEqual(gateway.calls, ["K1@p1"])

    def test_stream_error_before_first_chunk_switches_binding(self):
        _, gateway, runner = self.make(
            [error(503, "подождите"), sse(chunk({"content": "ок"}))]
        )
        stream = runner.stream("m", [{"role": "user", "content": "x"}])
        self.assertEqual(gateway.calls, ["K1@p1", "K1@p2"])
        self.assertEqual(stream.origin.proxy_name, "p2")
        list(stream)

    def test_partial_text_is_available_before_the_break(self):
        reply = httpx.Response(
            200,
            stream=BrokenStream(chunk({"content": "Первые"})),
            headers={"content-type": "text/event-stream"},
        )
        _, _, runner = self.make([reply])
        with (
            self.assertRaises(StreamBroken),
            runner.stream("m", [{"role": "user", "content": "x"}]) as stream,
        ):
            for item in stream:
                self.assertEqual(item.delta, "Первые")
                self.assertEqual(stream.text, "Первые")
            self.fail("обрыв должен был выбросить StreamBroken")

    def test_collect_on_closed_stream_is_safe(self):
        _, _, runner = self.make([sse(chunk({"content": "ок"}))])
        stream = runner.stream("m", [{"role": "user", "content": "x"}])
        list(stream)
        stream.close()
        self.assertEqual(stream.collect().text, "ок")
        self.assertEqual(stream.collect().text, "ок")


class LifecycleTests(RunnerCase):
    def test_runner_sends_to_the_address_it_names(self):
        _, gateway, runner = self.make(
            [error(502, "упало"), httpx.Response(200, json=completion())]
        )
        runner.complete("m", [{"role": "user", "content": "x"}])
        self.assertEqual([r["proxy_name"] for r in gateway.requests], ["p1", "p2"])

    def test_transport_is_reused_for_the_same_binding(self):
        _, gateway, runner = self.make(
            [
                httpx.Response(200, json=completion()),
                httpx.Response(200, json=completion()),
            ]
        )
        runner.complete("m", [{"role": "user", "content": "one"}])
        runner.complete("m", [{"role": "user", "content": "two"}])
        self.assertEqual(len(runner._transports), 1)
        self.assertEqual(len(gateway._clients), 1)

    def test_close_is_idempotent(self):
        _, _, runner = self.make([httpx.Response(200, json=completion())])
        runner.complete("m", [{"role": "user", "content": "x"}])
        runner.close()
        runner.close()
        self.assertEqual(runner._transports, {})

    def test_context_manager_closes(self):
        _, _, runner = self.make([httpx.Response(200, json=completion())])
        with runner as opened:
            self.assertIs(opened, runner)
        self.assertEqual(runner._transports, {})

    def test_runner_reports_its_size(self):
        _, _, runner = self.make([])
        self.assertIn("провайдеров 1", str(runner))
        self.assertIn("моделей 1", str(runner))


if __name__ == "__main__":
    unittest.main()
