"""Проверки реестра, связок (ключ, прокси) и матрицы отказов.

Запуск: python -m unittest test_model_registry -v
"""

from __future__ import annotations

import shutil
import tempfile
import time
import unittest
from pathlib import Path

import httpx

from llm_rotator import (
    ConfigError,
    Failure,
    Gateway,
    GatewayError,
    Key,
    Model,
    Provider,
    Proxy,
    Registry,
    SessionState,
    classify,
    load_registry,
)
from llm_rotator import discovery as md


def make_registry(
    proxies: tuple[str, ...] = ("direct", "p1", "p2"),
    key_proxies: tuple[tuple[str, tuple[str, ...]], ...] = (
        ("K1", ("p1", "p2")),
        ("K2", ("p1",)),
    ),
    extra_providers: dict[str, str] | None = None,
    model_providers: tuple[str, ...] = ("p",),
    key_owner: dict[str, str] | None = None,
) -> Registry:
    proxy_map = {
        name: Proxy(name=name, address="" if name == "direct" else f"http://{name}:1")
        for name in proxies
    }
    providers = {
        "p": Provider(
            name="p",
            api_style="openai",
            base_url="https://api.test/v1",
            models_url="https://api.test/v1/models",
        )
    }
    for name, base in (extra_providers or {}).items():
        providers[name] = Provider(
            name=name,
            api_style="openai",
            base_url=base,
            models_url=f"{base}/models",
        )
    keys = {
        name: Key(
            name=name,
            provider=(key_owner or {}).get(name, "p"),
            proxies=tuple(ps),
        )
        for name, ps in key_proxies
    }
    return Registry(
        providers=providers,
        proxies=proxy_map,
        keys=keys,
        models={"m": Model(name="m", providers=model_providers)},
        env_keys={name: f"secret-{name}" for name in keys},
    )


class StubGateway(Gateway):
    """Gateway с подставным транспортом: ответы задаёт сценарий.

    Свой клиент на каждый прокси, иначе в вызовах не видно, через какой адрес
    ушёл запрос: попытки помечаются как 'КЛЮЧ@адрес'.
    """

    def __init__(self, registry: Registry, script: list, state=None):
        self.calls: list[str] = []
        self._script = list(script)
        self._stub_clients: dict[str, httpx.Client] = {}
        # Значение ключа в логи не попадает: в попытках видно только имя.
        self._by_value = {value: name for name, value in registry.env_keys.items()}
        super().__init__(registry=registry, state=state)

    def client(self, proxy: Proxy) -> httpx.Client:
        client = self._stub_clients.get(proxy.name)
        if client is None:

            def handle(
                request: httpx.Request, name: str = proxy.name
            ) -> httpx.Response:
                value = request.headers.get("Authorization", "").removeprefix("Bearer ")
                self.calls.append(f"{self._by_value.get(value, '?')}@{name}")
                item = self._script.pop(0) if self._script else resp(200, {"ok": True})
                if isinstance(item, Exception):
                    raise item
                return item

            client = httpx.Client(transport=httpx.MockTransport(handle))
            self._stub_clients[proxy.name] = client
        return client

    def close(self) -> None:
        for client in self._stub_clients.values():
            client.close()
        self._stub_clients.clear()
        super().close()


def resp(status: int, body: object = None, text: str = "") -> httpx.Response:
    if text:
        return httpx.Response(status, text=text)
    return httpx.Response(status, json=body if body is not None else {})


KEY_ERROR = {"error": {"message": "invalid api key", "type": "invalid_request_error"}}
NO_SUCH_MODEL = {
    "error": {"message": "model not found", "type": "invalid_request_error"}
}
URL = "https://api.test/v1/models"


def fetch(gw: Gateway, registry: Registry, providers: tuple[str, ...] = ("p",)):
    return gw.request_routes(registry.routes("m"), "GET", URL)


class ClassifyTests(unittest.TestCase):
    def test_structured_403_is_key_problem(self):
        self.assertIs(classify(resp(403, KEY_ERROR)), Failure.KEY)

    def test_unstructured_403_is_proxy_problem(self):
        self.assertIs(classify(resp(403, text="error code: 1010")), Failure.PROXY)
        self.assertIs(classify(resp(403, text="<html>blocked</html>")), Failure.PROXY)

    def test_google_style_array_error_still_counts_as_provider_error(self):
        body = [{"error": {"code": 403, "message": "permission denied"}}]
        self.assertIs(classify(resp(403, body)), Failure.KEY)

    def test_auth_statuses(self):
        for status in (401, 402):
            self.assertIs(classify(resp(status, KEY_ERROR)), Failure.KEY)

    def test_rate_limit_is_a_pause_not_a_dead_key(self):
        """429 не убивает ключ: он жив, просто подождать."""
        self.assertIs(classify(resp(429, KEY_ERROR)), Failure.QUOTA)

    def test_quota_403_is_a_pause_too(self):
        body = {"error": {"message": "Quota exceeded for your project"}}
        self.assertIs(classify(resp(403, body)), Failure.QUOTA)

    def test_structured_404_is_provider_problem(self):
        self.assertIs(classify(resp(404, NO_SUCH_MODEL)), Failure.PROVIDER)

    def test_unstructured_404_is_fatal(self):
        self.assertIs(classify(resp(404, text="<html>404</html>")), Failure.FATAL)

    def test_gateway_errors_blame_the_address(self):
        for status in (502, 503, 504, 522):
            self.assertIs(classify(resp(status)), Failure.PROXY)

    def test_provider_own_errors_blame_the_model(self):
        """500 приходит от самого провайдера: адрес ни при чём."""
        for status in (500, 501, 505):
            self.assertIs(classify(resp(status)), Failure.MODEL)

    def test_client_errors_are_fatal(self):
        for status in (400, 422):
            self.assertIs(classify(resp(status, KEY_ERROR)), Failure.FATAL)


class BindingTests(unittest.TestCase):
    """Ключ ходит только через свои адреса: связку нельзя разобрать на части."""

    def test_key_failure_leaves_the_key_entirely(self):
        reg = make_registry(key_proxies=(("K1", ("p1", "p2")), ("K2", ("direct",))))
        gw = StubGateway(reg, [resp(401, KEY_ERROR), resp(200, {"ok": 1})])
        fetch(gw, reg)
        self.assertEqual(gw.calls, ["K1@p1", "K2@direct"])
        self.assertEqual(gw.state.bad_keys, {"K1"})
        self.assertEqual(gw.state.bad_proxies, set())

    def test_next_key_never_borrows_previous_key_proxies(self):
        reg = make_registry(key_proxies=(("K1", ("p1",)), ("K2", ("p2",))))
        gw = StubGateway(reg, [resp(401, KEY_ERROR), resp(401, KEY_ERROR)])
        with self.assertRaises(GatewayError):
            fetch(gw, reg)
        self.assertEqual(gw.calls, ["K1@p1", "K2@p2"])

    def test_bad_proxy_keeps_same_key(self):
        reg = make_registry(key_proxies=(("K1", ("p1", "p2")),))
        gw = StubGateway(reg, [resp(503), resp(200, {"ok": 1})])
        fetch(gw, reg)
        self.assertEqual(gw.calls, ["K1@p1", "K1@p2"])
        self.assertEqual(gw.state.bad_proxies, {"p1"})
        self.assertEqual(gw.state.bad_keys, set())

    def test_first_proxy_is_tried_first(self):
        reg = make_registry(key_proxies=(("K1", ("p1", "p2")),))
        gw = StubGateway(reg, [resp(200, {"ok": 1})])
        fetch(gw, reg)
        self.assertEqual(gw.calls, ["K1@p1"])

    def test_two_broken_proxies_do_not_burn_second_key(self):
        reg = make_registry(key_proxies=(("K1", ("p1", "p2")), ("K2", ("direct",))))
        gw = StubGateway(reg, [resp(502), resp(503), resp(200, {"ok": 1})])
        fetch(gw, reg)
        self.assertEqual(gw.calls, ["K1@p1", "K1@p2", "K2@direct"])
        self.assertEqual(gw.state.bad_proxies, {"p1", "p2"})

    def test_500_does_not_mark_the_address_bad(self):
        """Главное: 500 не выводит адрес из строя, обход идёт к провайдеру."""
        reg = make_registry(
            key_proxies=(("K1", ("p1", "p2")), ("K2", ("p2",))),
            key_owner={"K2": "q"},
            extra_providers={"q": "https://q.test"},
            model_providers=("p", "q"),
        )
        gw = StubGateway(reg, [resp(500), resp(200, {"ok": 1})])
        fetch(gw, reg)
        self.assertEqual(gw.calls, ["K1@p1", "K2@p2"])
        self.assertEqual(gw.state.bad_proxies, set())
        self.assertEqual(gw.state.unavailable, set())

    def test_direct_never_marked_bad(self):
        reg = make_registry(key_proxies=(("K1", ("direct", "p1")),))
        gw = StubGateway(reg, [httpx.ConnectError("boom"), resp(200, {"ok": 1})])
        fetch(gw, reg)
        self.assertNotIn("direct", gw.state.bad_proxies)
        self.assertEqual(gw.calls, ["K1@direct", "K1@p1"])

    def test_transport_error_is_proxy_problem(self):
        reg = make_registry(key_proxies=(("K1", ("p1", "p2")),))
        gw = StubGateway(reg, [httpx.ConnectTimeout("slow"), resp(200, {"ok": 1})])
        fetch(gw, reg)
        self.assertEqual(gw.state.bad_proxies, {"p1"})

    def test_unstructured_403_bans_address_not_key(self):
        reg = make_registry(key_proxies=(("K1", ("p1", "p2")),))
        gw = StubGateway(
            reg, [resp(403, text="error code: 1010"), resp(200, {"ok": 1})]
        )
        fetch(gw, reg)
        self.assertEqual(gw.calls, ["K1@p1", "K1@p2"])
        self.assertEqual(gw.state.bad_proxies, {"p1"})
        self.assertEqual(gw.state.bad_keys, set())

    def test_rate_limit_cools_down_instead_of_banning(self):
        reg = make_registry(key_proxies=(("K1", ("p1",)), ("K2", ("p2",))))
        state = SessionState()
        gw = StubGateway(reg, [resp(429, KEY_ERROR), resp(200, {"ok": 1})], state=state)
        gw.cooldown = 60
        fetch(gw, reg)
        self.assertEqual(gw.state.bad_keys, set())
        self.assertIn("K1", gw.state.cooldown_keys)
        self.assertFalse(gw.state.key_available("K1"))
        self.assertEqual(gw.calls, ["K1@p1", "K2@p2"])

    def test_cooldown_expires(self):
        state = SessionState()
        state.cool_down("K1", 0.01)
        self.assertFalse(state.key_available("K1"))
        time.sleep(0.05)
        self.assertTrue(state.key_available("K1"))

    def test_bad_state_is_not_retried_across_calls(self):
        reg = make_registry(key_proxies=(("K1", ("p1",)), ("K2", ("p2",))))
        gw = StubGateway(
            reg, [resp(401, KEY_ERROR), resp(200, {"ok": 1}), resp(200, {"ok": 2})]
        )
        fetch(gw, reg)
        fetch(gw, reg)
        self.assertEqual(gw.calls, ["K1@p1", "K2@p2", "K2@p2"])

    def test_fatal_stops_rotation(self):
        reg = make_registry(key_proxies=(("K1", ("p1", "p2")), ("K2", ("p2",))))
        gw = StubGateway(reg, [resp(400, KEY_ERROR)])
        with self.assertRaises(GatewayError) as ctx:
            fetch(gw, reg)
        self.assertEqual(gw.calls, ["K1@p1"])
        self.assertIn("invalid api key", str(ctx.exception))

    def test_exhausted_rotation_reports_all_attempts(self):
        reg = make_registry(key_proxies=(("K1", ("p1",)), ("K2", ("p2",))))
        gw = StubGateway(reg, [resp(401, KEY_ERROR), resp(401, KEY_ERROR)])
        with self.assertRaises(GatewayError) as ctx:
            fetch(gw, reg)
        message = str(ctx.exception)
        self.assertIn("K1", message)
        self.assertIn("K2", message)
        self.assertIn("HTTP 401", message)

    def test_no_attempts_when_everything_blacklisted(self):
        reg = make_registry(key_proxies=(("K1", ("p1",)), ("K2", ("p2",))))
        gw = StubGateway(reg, [])
        gw.state.bad_keys.update({"K1", "K2"})
        with self.assertRaises(GatewayError) as ctx:
            fetch(gw, reg)
        self.assertIn("попыток не было", str(ctx.exception))


class ProviderFailoverTests(unittest.TestCase):
    """Провайдеры перебираются только по явному списку модели."""

    def two_providers(self):
        return make_registry(
            proxies=("p1", "p2"),
            key_proxies=(("K1", ("p1",)), ("Q1", ("p2",))),
            extra_providers={"q": "https://api.q/v1"},
            model_providers=("p", "q"),
            key_owner={"Q1": "q"},
        )

    def test_404_switches_to_next_provider(self):
        reg = self.two_providers()
        gw = StubGateway(reg, [resp(404, NO_SUCH_MODEL), resp(200, {"ok": 1})])
        fetch(gw, reg)
        self.assertEqual(gw.calls, ["K1@p1", "Q1@p2"])
        self.assertIn(("p", "m"), gw.state.unavailable)

    def test_404_skips_remaining_keys_of_that_provider(self):
        reg = make_registry(
            proxies=("p1", "p2"),
            key_proxies=(("K1", ("p1",)), ("K2", ("p2",)), ("Q1", ("p1",))),
            extra_providers={"q": "https://api.q/v1"},
            model_providers=("p", "q"),
            key_owner={"Q1": "q"},
        )
        gw = StubGateway(reg, [resp(404, NO_SUCH_MODEL), resp(200, {"ok": 1})])
        fetch(gw, reg)
        # K2 не пробуем: 404 от провайдера закрывает его целиком, а не один ключ.
        self.assertEqual(gw.calls, ["K1@p1", "Q1@p1"])

    def test_provider_order_is_respected(self):
        reg = self.two_providers()
        gw = StubGateway(reg, [resp(200, {"ok": 1})])
        fetch(gw, reg)
        self.assertEqual(gw.calls, ["K1@p1"])

    def test_disabled_provider_is_not_retried_across_calls(self):
        reg = self.two_providers()
        gw = StubGateway(reg, [resp(404, NO_SUCH_MODEL), resp(200, {"ok": 1})])
        fetch(gw, reg)
        fetch(gw, reg)
        self.assertEqual(gw.calls, ["K1@p1", "Q1@p2", "Q1@p2"])

    def test_unavailable_provider_is_reported(self):
        reg = self.two_providers()
        gw = StubGateway(reg, [resp(404, NO_SUCH_MODEL), resp(401, KEY_ERROR)])
        with self.assertRaises(GatewayError) as ctx:
            fetch(gw, reg)
        message = str(ctx.exception)
        self.assertIn("модели нет (помечен после 404)", message)
        self.assertIn("HTTP 401", message)


class AuthNoneTests(unittest.TestCase):
    """У провайдера без ключей заголовок авторизации уходить не должен.

    Пустой «Bearer » — невалидное значение заголовка, и запрос с ним не уходит
    вообще: локальный Ollama из-за этого не отвечал на список моделей.
    """

    def registry(self, auth: str) -> Registry:
        return Registry(
            providers={
                "p": Provider(
                    name="p",
                    api_style="openai",
                    base_url="http://127.0.0.1:11434/v1",
                    models_url="http://127.0.0.1:11434/v1/models",
                    auth=auth,
                )
            },
            proxies={"direct": Proxy(name="direct", address="")},
            keys={"K1": Key(name="K1", provider="p", proxies=("direct",))},
            models={"m": Model(name="m", providers=("p",))},
            env_keys={} if auth == "none" else {"K1": "secret-K1"},
        )

    def sent_headers(self, auth: str) -> httpx.Headers:
        seen: list[httpx.Headers] = []

        class Gw(Gateway):
            def client(self, proxy):
                def handle(request: httpx.Request) -> httpx.Response:
                    seen.append(request.headers)
                    return resp(200, {"data": [{"id": "m"}]})

                return httpx.Client(transport=httpx.MockTransport(handle))

        gw = Gw(registry=self.registry(auth))
        try:
            self.assertEqual(gw.list_models("p"), ["m"])
        finally:
            gw.close()
        return seen[0]

    def test_bearer_header_is_dropped_when_auth_is_none(self):
        self.assertNotIn("authorization", self.sent_headers("none"))

    def test_bearer_header_is_sent_when_there_is_a_key(self):
        self.assertIn("authorization", self.sent_headers("bearer"))


class ModelListTests(unittest.TestCase):
    def test_strips_resource_prefix(self):
        reg = make_registry(key_proxies=(("K1", ("p1",)),))
        reg.providers["p"] = Provider(
            name="p",
            api_style="openai",
            base_url="https://api.test/v1",
            models_url="https://api.test/v1/models",
            model_id_prefix="models/",
        )
        gw = StubGateway(
            reg, [resp(200, {"data": [{"id": "models/a"}, {"id": "models/b"}]})]
        )
        self.assertEqual(gw.list_models("p"), ["a", "b"])

    def test_list_models_uses_provider_bindings(self):
        reg = make_registry(key_proxies=(("K1", ("p1", "p2")),))
        gw = StubGateway(reg, [resp(503), resp(200, {"data": [{"id": "a"}]})])
        self.assertEqual(gw.list_models("p"), ["a"])
        self.assertEqual(gw.calls, ["K1@p1", "K1@p2"])


class ValidationTests(unittest.TestCase):
    def setUp(self):
        self.dir = Path(tempfile.mkdtemp())
        (self.dir / "models").mkdir()
        (self.dir / ".env").write_text("K1=one\nK2=two\n", encoding="utf-8")
        self.write(
            "providers.toml",
            """
[providers.p]
provider_name = "p"
base_url = "https://api.test/v1"
models_url = "https://api.test/v1/models"
""",
        )
        self.write(
            "proxies.toml",
            """
[proxies.direct]
proxy_name = "direct"
proxy_address = ""
""",
        )
        self.write(
            "keys.toml",
            """
[[keys]]
key_name = "K1"
provider = "p"
proxies = ["direct"]
""",
        )
        self.write(
            "models.toml",
            """
[[models]]
model_name = "m"
providers = ["p"]
""",
        )

    def tearDown(self):
        shutil.rmtree(self.dir, ignore_errors=True)

    def write(self, name: str, text: str) -> None:
        (self.dir / "models" / name).write_text(text, encoding="utf-8")

    def load(self) -> Registry:
        return load_registry(self.dir)

    def test_valid_config_loads(self):
        reg = self.load()
        self.assertEqual(len(reg.models), 1)
        route = reg.route("m", "p")
        self.assertEqual(route.key_names, ("K1",))
        self.assertIsNone(route.bindings[0].proxy.httpx_proxy)
        self.assertEqual(str(route.bindings[0]), "K1@direct")

    def test_unknown_provider_lists_valid_names(self):
        self.write(
            "models.toml",
            '[[models]]\nmodel_name = "m"\nproviders = ["nope"]\n',
        )
        with self.assertRaises(ConfigError) as ctx:
            self.load()
        self.assertIn("'p'", str(ctx.exception))

    def test_max_output_tokens_is_read_when_present(self):
        self.write(
            "models.toml",
            '[[models]]\nmodel_name = "m"\nproviders = ["p"]\n'
            "max_output_tokens = 999\n",
        )
        self.assertEqual(self.load().model("m").max_output_tokens, 999)

    def test_max_output_tokens_defaults_to_none(self):
        """Нет поля — клиент не шлёт max_tokens и не спорит с провайдером."""
        self.assertIsNone(self.load().model("m").max_output_tokens)

    def test_chat_history_pairs_is_read_on_model(self):
        """Ротация хранит поле, но не толкует его — это дело вызывающего кода."""
        self.write(
            "models.toml",
            '[[models]]\nmodel_name = "m"\nproviders = ["p"]\nchat_history_pairs = 5\n',
        )
        self.assertEqual(self.load().model("m").chat_history_pairs, 5)

    def test_max_output_tokens_must_be_a_positive_number(self):
        for bad in ("0", "-5", '"999"', "true"):
            with self.subTest(bad=bad):
                self.write(
                    "models.toml",
                    '[[models]]\nmodel_name = "m"\nproviders = ["p"]\n'
                    f"max_output_tokens = {bad}\n",
                )
                with self.assertRaises(ConfigError) as ctx:
                    self.load()
                self.assertIn("max_output_tokens", str(ctx.exception))

    def test_unknown_proxy_in_key_lists_valid_names(self):
        self.write(
            "keys.toml",
            '[[keys]]\nkey_name = "K1"\nprovider = "p"\nproxies = ["nope"]\n',
        )
        with self.assertRaises(ConfigError) as ctx:
            self.load()
        self.assertIn("'direct'", str(ctx.exception))

    def test_key_cannot_point_at_unknown_provider(self):
        self.write(
            "keys.toml",
            '[[keys]]\nkey_name = "K1"\nprovider = "nope"\nproxies = ["direct"]\n',
        )
        with self.assertRaises(ConfigError) as ctx:
            self.load()
        self.assertIn("'p'", str(ctx.exception))

    def test_name_field_must_match_table_key(self):
        self.write(
            "proxies.toml",
            '[proxies.direct]\nproxy_name = "drect"\nproxy_address = ""\n',
        )
        with self.assertRaises(ConfigError) as ctx:
            self.load()
        self.assertIn("не совпадает с ключом таблицы", str(ctx.exception))

    def test_direct_proxy_is_mandatory(self):
        self.write(
            "proxies.toml",
            '[proxies.p1]\nproxy_name = "p1"\nproxy_address = "http://h"\n',
        )
        with self.assertRaises(ConfigError) as ctx:
            self.load()
        self.assertIn("direct", str(ctx.exception))

    def test_direct_proxy_must_be_addressless(self):
        self.write(
            "proxies.toml",
            '[proxies.direct]\nproxy_name = "direct"\nproxy_address = "http://h"\n',
        )
        with self.assertRaises(ConfigError):
            self.load()

    def test_address_without_scheme_gets_http(self):
        self.write(
            "proxies.toml",
            '[proxies.direct]\nproxy_name = "direct"\nproxy_address = ""\n'
            '[proxies.p1]\nproxy_name = "p1"\nproxy_address = "127.0.0.1:1091"\n',
        )
        proxy = self.load().proxy("p1")
        self.assertEqual(proxy.httpx_proxy, "http://127.0.0.1:1091")
        self.assertEqual(proxy.proxy_type, "http")
        self.assertFalse(proxy.is_socks)

    def test_direct_proxy_has_no_address_for_httpx(self):
        self.write(
            "proxies.toml",
            '[proxies.direct]\nproxy_name = "direct"\nproxy_address = ""\n',
        )
        self.assertIsNone(self.load().proxy("direct").httpx_proxy)

    def test_socks_type_is_read(self):
        self.write(
            "proxies.toml",
            '[proxies.direct]\nproxy_name = "direct"\nproxy_address = ""\n'
            '[proxies.s1]\nproxy_name = "s1"\nproxy_address = "127.0.0.1:1080"\n'
            'proxy_type = "socks5"\n',
        )
        proxy = self.load().proxy("s1")
        self.assertEqual(proxy.httpx_proxy, "socks5://127.0.0.1:1080")
        self.assertTrue(proxy.is_socks)

    def test_scheme_in_address_replaces_type(self):
        self.write(
            "proxies.toml",
            '[proxies.direct]\nproxy_name = "direct"\nproxy_address = ""\n'
            '[proxies.s1]\nproxy_name = "s1"\n'
            'proxy_address = "socks5h://proxy.local:1080"\n',
        )
        proxy = self.load().proxy("s1")
        self.assertEqual(proxy.proxy_type, "socks5h")
        self.assertEqual(proxy.httpx_proxy, "socks5h://proxy.local:1080")

    def test_unknown_proxy_type_lists_valid_names(self):
        self.write(
            "proxies.toml",
            '[proxies.direct]\nproxy_name = "direct"\nproxy_address = ""\n'
            '[proxies.s1]\nproxy_name = "s1"\nproxy_address = "127.0.0.1:1080"\n'
            'proxy_type = "socks4"\n',
        )
        with self.assertRaises(ConfigError) as ctx:
            self.load()
        message = str(ctx.exception)
        for name in ("http", "https", "socks5", "socks5h"):
            self.assertIn(f"'{name}'", message)

    def test_unknown_scheme_in_address_is_rejected(self):
        self.write(
            "proxies.toml",
            '[proxies.direct]\nproxy_name = "direct"\nproxy_address = ""\n'
            '[proxies.s1]\nproxy_name = "s1"\n'
            'proxy_address = "socks4://127.0.0.1:1080"\n',
        )
        with self.assertRaises(ConfigError) as ctx:
            self.load()
        self.assertIn("socks4", str(ctx.exception))

    def test_scheme_conflicting_with_type_is_rejected(self):
        self.write(
            "proxies.toml",
            '[proxies.direct]\nproxy_name = "direct"\nproxy_address = ""\n'
            '[proxies.s1]\nproxy_name = "s1"\nproxy_address = "http://127.0.0.1:1080"\n'
            'proxy_type = "socks5"\n',
        )
        with self.assertRaises(ConfigError) as ctx:
            self.load()
        self.assertIn("противоречие", str(ctx.exception))

    def test_duplicate_key_name_rejected(self):
        self.write(
            "keys.toml",
            '[[keys]]\nkey_name = "K1"\nprovider = "p"\nproxies = ["direct"]\n'
            '[[keys]]\nkey_name = "K1"\nprovider = "p"\nproxies = ["direct"]\n',
        )
        with self.assertRaises(ConfigError) as ctx:
            self.load()
        self.assertIn("дубликат", str(ctx.exception))

    def test_duplicate_model_name_rejected(self):
        self.write(
            "models.toml",
            '[[models]]\nmodel_name = "m"\nproviders = ["p"]\n'
            '[[models]]\nmodel_name = "m"\nproviders = ["p"]\n',
        )
        with self.assertRaises(ConfigError) as ctx:
            self.load()
        self.assertIn("дубликат", str(ctx.exception))

    def test_model_name_may_be_described_once_for_two_providers(self):
        self.write(
            "providers.toml",
            """
[providers.p]
provider_name = "p"
base_url = "https://a/v1"
models_url = "https://a/v1/models"
[providers.q]
provider_name = "q"
base_url = "https://b/v1"
models_url = "https://b/v1/models"
""",
        )
        self.write(
            "keys.toml",
            '[[keys]]\nkey_name = "K1"\nprovider = "p"\nproxies = ["direct"]\n'
            '[[keys]]\nkey_name = "K2"\nprovider = "q"\nproxies = ["direct"]\n',
        )
        self.write(
            "models.toml", '[[models]]\nmodel_name = "m"\nproviders = ["p", "q"]\n'
        )
        reg = self.load()
        self.assertEqual(len(reg.models), 1)
        self.assertEqual([r.provider.name for r in reg.routes("m")], ["p", "q"])

    def test_missing_key_is_error_when_provider_in_use(self):
        (self.dir / ".env").write_text("K1=one\n", encoding="utf-8")
        self.write(
            "keys.toml",
            '[[keys]]\nkey_name = "K1"\nprovider = "p"\nproxies = ["direct"]\n'
            '[[keys]]\nkey_name = "K2"\nprovider = "p"\nproxies = ["direct"]\n',
        )
        with self.assertRaises(ConfigError) as ctx:
            self.load()
        self.assertIn("K2", str(ctx.exception))

    def test_missing_key_is_warning_when_provider_unused(self):
        (self.dir / ".env").write_text("K1=one\n", encoding="utf-8")
        self.write(
            "keys.toml",
            '[[keys]]\nkey_name = "K1"\nprovider = "p"\nproxies = ["direct"]\n'
            '[[keys]]\nkey_name = "K2"\nprovider = "q"\nproxies = ["direct"]\n',
        )
        self.write(
            "providers.toml",
            '[providers.p]\nprovider_name = "p"\nbase_url = "https://a/v1"\n'
            'models_url = "https://a/v1/models"\n'
            '[providers.q]\nprovider_name = "q"\nbase_url = "https://b/v1"\n'
            'models_url = "https://b/v1/models"\n',
        )
        self.write("models.toml", '[[models]]\nmodel_name = "m"\nproviders = ["p"]\n')
        reg = self.load()
        self.assertTrue(any("K2" in w for w in reg.warnings))

    def test_disabled_model_is_excluded_from_routing(self):
        self.write(
            "models.toml",
            '[[models]]\nmodel_name = "m"\nproviders = ["p"]\nenabled = false\n',
        )
        with self.assertRaises(ConfigError) as ctx:
            self.load().route("m", "p")
        self.assertIn("отключена", str(ctx.exception))

    def test_route_without_credentials_explains_why(self):
        self.write(
            "keys.toml",
            '[[keys]]\nkey_name = "K1"\nprovider = "p"\nproxies = ["direct"]\n',
        )
        (self.dir / ".env").write_text("OTHER=x\n", encoding="utf-8")
        with self.assertRaises(ConfigError) as ctx:
            self.load()
        self.assertIn(".env", str(ctx.exception))

    def test_route_without_keys_in_toml_explains_why(self):
        self.write("keys.toml", "")
        reg = self.load()
        with self.assertRaises(ConfigError) as ctx:
            reg.route("m", "p")
        self.assertIn("ни одного ключа", str(ctx.exception))

    def test_lookup_error_hints_similar_models(self):
        self.write(
            "models.toml",
            '[[models]]\nmodel_name = "model-x"\nproviders = ["p"]\n',
        )
        with self.assertRaises(ConfigError) as ctx:
            self.load().route("m", "p")
        self.assertIn("Похожие", str(ctx.exception))

    def test_broken_toml_reports_file(self):
        self.write("proxies.toml", "[providers.direct\n")
        with self.assertRaises(ConfigError) as ctx:
            self.load()
        self.assertIn("proxies.toml", str(ctx.exception))

    def test_unknown_auth_lists_valid_names(self):
        self.write(
            "providers.toml",
            """
[providers.p]
provider_name = "p"
base_url = "https://api.test/v1"
models_url = "https://api.test/v1/models"
auth = "magic"
""",
        )
        with self.assertRaises(ConfigError) as ctx:
            self.load()
        self.assertIn("'bearer'", str(ctx.exception))
        self.assertIn("'none'", str(ctx.exception))

    def test_auth_none_needs_no_env_value(self):
        self.write(
            "providers.toml",
            """
[providers.p]
provider_name = "p"
base_url = "http://127.0.0.1:11434/v1"
models_url = "http://127.0.0.1:11434/v1/models"
auth = "none"
""",
        )
        self.write(
            "keys.toml",
            '[[keys]]\nkey_name = "K1"\nprovider = "p"\nproxies = ["direct"]\n',
        )
        (self.dir / ".env").write_text("OTHER=x\n", encoding="utf-8")
        reg = self.load()
        binding = reg.route("m", "p").bindings[0]
        self.assertEqual(binding.key_value, "")
        self.assertEqual(str(binding), "K1@direct")
        self.assertEqual(reg.warnings, [])

    def test_auth_still_requires_env_value(self):
        (self.dir / ".env").write_text("OTHER=x\n", encoding="utf-8")
        with self.assertRaises(ConfigError) as ctx:
            self.load()
        self.assertIn(".env", str(ctx.exception))

    def test_prompt_profile_is_not_validated_here(self):
        """Какие профили бывают, знает вызывающий код, а не ротация."""
        self.write(
            "models.toml",
            '[[models]]\nmodel_name = "m"\nproviders = ["p"]\n'
            'prompt_profile = "huge"\n',
        )
        self.assertEqual(self.load().model("m").prompt_profile, "huge")

    def test_prompt_profile_defaults_to_empty(self):
        self.assertEqual(self.load().model("m").prompt_profile, "")

    def test_prompt_profile_is_read(self):
        self.write(
            "models.toml",
            '[[models]]\nmodel_name = "m"\nproviders = ["p"]\n'
            'prompt_profile = "core"\n',
        )
        self.assertEqual(self.load().model("m").prompt_profile, "core")

    def test_extra_headers_are_read(self):
        self.write(
            "providers.toml",
            """
[providers.p]
provider_name = "p"
base_url = "https://api.test/v1"
models_url = "https://api.test/v1/models"

[providers.p.extra_headers]
HTTP-Referer = "https://example.org"
X-Title = "yulya-bot"
""",
        )
        provider = self.load().provider("p")
        self.assertEqual(
            provider.headers,
            {"HTTP-Referer": "https://example.org", "X-Title": "yulya-bot"},
        )

    def test_non_ascii_header_value_is_rejected(self):
        self.write(
            "providers.toml",
            """
[providers.p]
provider_name = "p"
base_url = "https://api.test/v1"
models_url = "https://api.test/v1/models"

[providers.p.extra_headers]
X-Title = "Юля"
""",
        )
        with self.assertRaises(ConfigError) as ctx:
            self.load()
        self.assertIn("X-Title", str(ctx.exception))

    def test_extra_headers_must_be_strings(self):
        self.write(
            "providers.toml",
            """
[providers.p]
provider_name = "p"
base_url = "https://api.test/v1"
models_url = "https://api.test/v1/models"

[providers.p.extra_headers]
X-Title = 7
""",
        )
        with self.assertRaises(ConfigError) as ctx:
            self.load()
        self.assertIn("X-Title", str(ctx.exception))


class SharedProxyWarningTests(unittest.TestCase):
    def setUp(self):
        self.dir = Path(tempfile.mkdtemp())
        (self.dir / "models").mkdir()
        (self.dir / ".env").write_text("K1=one\nK2=two\n", encoding="utf-8")
        self.write(
            "providers.toml",
            '[providers.p]\nprovider_name = "p"\nbase_url = "https://a/v1"\n'
            'models_url = "https://a/v1/models"\n',
        )
        self.write("models.toml", '[[models]]\nmodel_name = "m"\nproviders = ["p"]\n')

    def tearDown(self):
        shutil.rmtree(self.dir, ignore_errors=True)

    def write(self, name: str, text: str) -> None:
        (self.dir / "models" / name).write_text(text, encoding="utf-8")

    def keys_and_proxies(self, proxy_body: str) -> Registry:
        self.write(
            "proxies.toml",
            '[proxies.direct]\nproxy_name = "direct"\nproxy_address = ""\n'
            + proxy_body,
        )
        self.write(
            "keys.toml",
            '[[keys]]\nkey_name = "K1"\nprovider = "p"\nproxies = ["direct", "shared"]\n'
            '[[keys]]\nkey_name = "K2"\nprovider = "p"\nproxies = ["shared"]\n',
        )
        return load_registry(self.dir)

    def test_shared_plain_proxy_warns(self):
        reg = self.keys_and_proxies(
            '[proxies.shared]\nproxy_name = "shared"\nproxy_address = "http://s:1"\n'
        )
        self.assertTrue(any("shared" in w for w in reg.warnings))

    def test_shared_rotating_proxy_is_fine(self):
        reg = self.keys_and_proxies(
            '[proxies.shared]\nproxy_name = "shared"\nproxy_address = "http://s:1"\n'
            "rotates = true\n"
        )
        self.assertFalse(any("абуз" in w or "выглядит как" in w for w in reg.warnings))

    def test_shared_direct_warns_about_single_ip(self):
        self.write(
            "proxies.toml",
            '[proxies.direct]\nproxy_name = "direct"\nproxy_address = ""\n',
        )
        self.write(
            "keys.toml",
            '[[keys]]\nkey_name = "K1"\nprovider = "p"\nproxies = ["direct"]\n'
            '[[keys]]\nkey_name = "K2"\nprovider = "p"\nproxies = ["direct"]\n',
        )
        reg = load_registry(self.dir)
        self.assertTrue(any("с одного IP" in w for w in reg.warnings))

    def test_unused_proxy_warns(self):
        self.write(
            "proxies.toml",
            '[proxies.direct]\nproxy_name = "direct"\nproxy_address = ""\n'
            '[proxies.spare]\nproxy_name = "spare"\nproxy_address = "http://s:1"\n',
        )
        self.write(
            "keys.toml",
            '[[keys]]\nkey_name = "K1"\nprovider = "p"\nproxies = ["direct"]\n',
        )
        reg = load_registry(self.dir)
        self.assertTrue(any("spare" in w for w in reg.warnings))


class MergeTests(unittest.TestCase):
    def setUp(self):
        self.dir = Path(tempfile.mkdtemp())
        self.models = self.dir / "models.toml"
        self.models.write_text(
            '# шапка\n\n[[models]]\nmodel_name = "m"\nproviders = ["p"]\n',
            encoding="utf-8",
        )

    def tearDown(self):
        shutil.rmtree(self.dir, ignore_errors=True)

    def test_adds_only_missing_models(self):
        reg = make_registry(extra_providers={"q": "https://q/v1"})
        added = md.merge_into_models(
            reg, {"p": ("a", "m"), "q": ("b",)}, path=self.models, dry_run=True
        )
        self.assertEqual([m.name for m in added], ["a", "b"])

    def test_dry_run_writes_nothing(self):
        before = self.models.read_text(encoding="utf-8")
        md.merge_into_models(
            make_registry(), {"p": ("a",)}, path=self.models, dry_run=True
        )
        self.assertEqual(self.models.read_text(encoding="utf-8"), before)

    def test_existing_providers_are_never_extended(self):
        md.merge_into_models(
            make_registry(extra_providers={"q": "https://q/v1"}),
            {"p": ("m", "a"), "q": ("m", "b")},
            path=self.models,
        )
        text = self.models.read_text(encoding="utf-8")
        self.assertIn("# шапка", text)
        body = text.split("[[models]]")[1]
        self.assertIn('model_name = "m"', body)
        self.assertIn('providers = ["p"]', body)

    def test_written_file_is_parseable(self):
        import tomllib

        md.merge_into_models(make_registry(), {"p": ("a",)}, path=self.models)
        data = tomllib.loads(self.models.read_text(encoding="utf-8"))
        names = [t["model_name"] for t in data["models"]]
        self.assertEqual(names, ["m", "a"])
        self.assertEqual(data["models"][1]["providers"], ["p"])


class CacheTests(unittest.TestCase):
    def setUp(self):
        self.dir = Path(tempfile.mkdtemp())
        self.cache = self.dir / "models.discovered.toml"

    def tearDown(self):
        shutil.rmtree(self.dir, ignore_errors=True)

    def test_fresh_cache_is_used(self):
        self.cache.write_text(
            'generated_at = "2099-01-01T00:00:00+00:00"\n'
            '[discovered.p]\nmodel_names = ["a", "b"]\n',
            encoding="utf-8",
        )
        self.assertTrue(md.cache_is_fresh(self.cache))
        self.assertEqual(md.load_discovered(self.cache), {"p": ("a", "b")})

    def test_stale_cache_is_not_fresh(self):
        self.cache.write_text(
            'generated_at = "2000-01-01T00:00:00+00:00"\n'
            '[discovered.p]\nmodel_names = ["a"]\n',
            encoding="utf-8",
        )
        self.assertFalse(md.cache_is_fresh(self.cache))

    def test_missing_cache_is_not_fresh(self):
        self.assertFalse(md.cache_is_fresh(self.cache))
        self.assertIsNone(md.cache_age(self.cache))
        self.assertEqual(md.load_discovered(self.cache), {})

    def test_corrupt_cache_is_ignored(self):
        self.cache.write_text("[[[broken", encoding="utf-8")
        self.assertEqual(md.load_discovered(self.cache), {})

    def test_failed_providers_are_not_written(self):
        text = md.render_cache(
            {
                "p": md.Discovered("p", ("a", "b")),
                "q": md.Discovered("q", (), error="boom"),
            }
        )
        self.assertIn("[discovered.p]", text)
        self.assertNotIn("[discovered.q]", text)

    def test_extra_providers_are_only_suggested(self):
        reg = make_registry(
            extra_providers={"q": "https://q/v1"}, model_providers=("p",)
        )
        extra = md.find_extra_providers(reg, {"q": ("m", "x")})
        self.assertEqual(extra, {"m": ["q"]})
        self.assertEqual(reg.models["m"].providers, ("p",))


if __name__ == "__main__":
    unittest.main(verbosity=2)
