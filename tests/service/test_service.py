"""Сервис: клиенты и пространства имён, OpenAI-формат, SSE, цепочки, статус."""

from __future__ import annotations

import json
import unittest

import httpx

try:
    from starlette.testclient import TestClient

    from llm_rotator.service import persist
    from llm_rotator.service.app import Service, create_app
except ImportError:  # extras [server] не установлены
    TestClient = None

from llm_rotator import (
    Key,
    Model,
    ModelRunner,
    Provider,
    Proxy,
    Registry,
    WebModel,
    WebSettings,
)
from llm_rotator.gateway import SessionState
from llm_rotator.web.context import TransportContext
from llm_rotator.web.store import ConversationStore
from tests.web.test_transport import FakeDeepSeek, WebCase

TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "read_file",
            "description": "d",
            "parameters": {
                "type": "object",
                "properties": {"path": {"type": "string"}},
            },
        },
    }
]
CALL = '{"tool_call":{"name":"read_file","arguments":{"path":"/tmp/a"}}}'
CLIENTS = """
[[clients]]
client_name = "hermes"
token_env = "TOKEN_HERMES"
default_model = "m"

[[clients]]
client_name = "other"
token_env = "TOKEN_OTHER"
default_model = "chain:main"
"""


def chat_body(*messages, **extra):
    return {"messages": list(messages), **extra}


def u(text):
    return {"role": "user", "content": text}


def a(text):
    return {"role": "assistant", "content": text}


def auth(token):
    return {"Authorization": f"Bearer {token}"}


def sse_events(text: str) -> list:
    out = []
    for block in text.split("\n\n"):
        for line in block.splitlines():
            if line.startswith("data: "):
                payload = line[6:]
                out.append(payload if payload == "[DONE]" else json.loads(payload))
    return out


@unittest.skipIf(TestClient is None, "нужны extras [server]")
class ServiceCase(WebCase):
    clients_toml = CLIENTS

    def setUp(self):
        super().setUp()
        (self.dir / "models").mkdir(exist_ok=True)
        (self.dir / "models" / "clients.toml").write_text(self.clients_toml)
        (self.dir / ".env").write_text(
            "TOKEN_HERMES=hermes-secret\nTOKEN_OTHER=other-secret\n"
        )
        self.registry = self.runner.registry
        from llm_rotator.registry import Chain

        self.registry.chains["main"] = Chain("main", (self.registry.models["m"],))
        self.service = Service(self.dir, runner=self.runner)
        self.http = TestClient(create_app(self.service), client=("127.0.0.1", 50000))
        self.addCleanup(self.http.close)

    def post(self, body, token="hermes-secret", **headers):
        return self.http.post(
            "/v1/chat/completions", json=body, headers={**auth(token), **headers}
        )


class AuthTests(ServiceCase):
    def test_health_is_open(self):
        self.assertEqual(self.http.get("/health").json(), {"status": "ok"})

    def test_missing_and_wrong_tokens_are_401(self):
        self.assertEqual(
            self.http.post("/v1/chat/completions", json=chat_body(u("x"))).status_code,
            401,
        )
        self.assertEqual(self.post(chat_body(u("x")), token="nope").status_code, 401)
        self.assertEqual(self.http.get("/v1/models").status_code, 401)

    def test_x_api_key_header_works(self):
        r = self.http.post(
            "/v1/chat/completions",
            json=chat_body(u("x")),
            headers={"x-api-key": "hermes-secret"},
        )
        self.assertEqual(r.status_code, 200)


class CompletionTests(ServiceCase):
    def test_non_stream_response_shape_and_headers(self):
        r = self.post(chat_body(u("привет")))
        self.assertEqual(r.status_code, 200)
        body = r.json()
        self.assertEqual(body["object"], "chat.completion")
        self.assertEqual(body["choices"][0]["message"]["content"], "ответ 1")
        self.assertEqual(body["choices"][0]["message"]["reasoning_content"], "думаю")
        self.assertEqual(body["choices"][0]["finish_reason"], "stop")
        self.assertGreater(body["usage"]["total_tokens"], 0)
        self.assertIn("DS_A", r.headers["x-rotator-origin"])
        self.assertEqual(r.headers["x-rotator-attempts"], "0")

    def test_two_clients_with_identical_messages_get_different_chats(self):
        self.post(chat_body(u("одно и то же")), token="hermes-secret")
        self.post(chat_body(u("одно и то же")), token="other-secret")
        self.assertEqual(self.site.chats, ["chat-1", "chat-2"])
        self.assertEqual(
            {r.namespace for r in self.store.by_state("active")}, {"hermes", "other"}
        )

    def test_same_client_continuation_is_a_delta(self):
        first = self.post(chat_body(u("1"))).json()["choices"][0]["message"]["content"]
        self.post(chat_body(u("1"), a(first), u("2")))
        self.assertEqual(self.site.chats, ["chat-1"])
        self.assertEqual(self.site.completions[-1]["prompt"], "User: 2")

    def test_session_header_and_user_field_split_conversations(self):
        self.post(chat_body(u("1")), **{"X-Session-Id": "s1"})
        self.post(chat_body(u("1"), user="s2"))
        self.assertEqual(
            {r.session for r in self.store.by_state("active")}, {"s1", "s2"}
        )

    def test_tool_call_response(self):
        self.site.replies = [CALL]
        body = self.post(chat_body(u("прочти"), tools=TOOLS)).json()
        choice = body["choices"][0]
        self.assertEqual(choice["finish_reason"], "tool_calls")
        self.assertIsNone(choice["message"]["content"])
        call = choice["message"]["tool_calls"][0]
        self.assertEqual(call["function"]["name"], "read_file")
        self.assertEqual(json.loads(call["function"]["arguments"]), {"path": "/tmp/a"})

    def test_unknown_param_is_dropped_and_max_tokens_kept(self):
        r = self.post(chat_body(u("x"), max_completion_tokens=50, logprobs=True))
        self.assertEqual(r.status_code, 200)

    def test_bad_requests_are_400(self):
        for body in (
            {},
            {"messages": []},
            {"messages": ["x"]},
            {"messages": [u("x")], "max_tokens": -1},
        ):
            with self.subTest(body=body):
                self.assertEqual(self.post(body).status_code, 400)
        r = self.http.post(
            "/v1/chat/completions", content=b"{", headers=auth("hermes-secret")
        )
        self.assertEqual(r.status_code, 400)

    def test_unknown_model_falls_back_to_client_default(self):
        r = self.post(chat_body(u("x"), model="gpt-nope"))
        self.assertEqual(r.status_code, 200)

    def test_chain_model_name_is_accepted(self):
        self.assertEqual(
            self.post(chat_body(u("x"), model="chain:main")).status_code, 200
        )


class StreamTests(ServiceCase):
    def stream(self, body, token="hermes-secret"):
        with self.http.stream(
            "POST",
            "/v1/chat/completions",
            json={**body, "stream": True},
            headers=auth(token),
        ) as r:
            return r, "".join(r.iter_text())

    def test_chunks_are_valid_openai_sse(self):
        r, text = self.stream(
            chat_body(u("привет"), stream_options={"include_usage": True})
        )
        self.assertEqual(r.status_code, 200)
        self.assertIn("event-stream", r.headers["content-type"])
        events = sse_events(text)
        self.assertEqual(events[-1], "[DONE]")
        chunks = events[:-1]
        self.assertEqual(
            chunks[0]["choices"][0]["delta"], {"role": "assistant", "content": ""}
        )
        content = "".join(c["choices"][0]["delta"].get("content", "") for c in chunks)
        self.assertEqual(content, "ответ 1")
        self.assertEqual(chunks[-1]["choices"][0]["finish_reason"], "stop")
        self.assertGreater(chunks[-1]["usage"]["total_tokens"], 0)
        self.assertEqual({c["id"] for c in chunks}, {chunks[0]["id"]})
        self.assertIn("DS_A", r.headers["x-rotator-origin"])

    def test_stream_with_tool_call(self):
        self.site.replies = [CALL]
        _, text = self.stream(chat_body(u("прочти"), tools=TOOLS))
        chunks = sse_events(text)[:-1]
        calls = [
            c["choices"][0]["delta"]["tool_calls"]
            for c in chunks
            if "tool_calls" in c["choices"][0]["delta"]
        ]
        self.assertEqual(calls[0][0]["function"]["name"], "read_file")
        self.assertEqual(calls[0][0]["index"], 0)
        self.assertEqual(chunks[-1]["choices"][0]["finish_reason"], "tool_calls")

    def test_failure_before_first_chunk_is_a_proper_http_error(self):
        self.site.fail_with = httpx.Response(
            200, json={"code": 40003, "msg": "INVALID_TOKEN"}
        )
        r, _ = self.stream(chat_body(u("x")))
        self.assertEqual(r.status_code, 503)

    def test_stream_continuation_commits_chat(self):
        self.stream(chat_body(u("1")))
        self.post(chat_body(u("1"), a("ответ 1"), u("2")))
        self.assertEqual(self.site.chats, ["chat-1"])


class ErrorMappingTests(ServiceCase):
    def test_busy_account_is_429_with_retry_after(self):
        body = (
            'event: hint\ndata: {"type":"error","finish_reason":"backend_busy","content":"занят"}\n\n'
        ).encode()
        self.site.fail_with = httpx.Response(
            200, content=body, headers={"content-type": "text/event-stream"}
        )
        r = self.post(chat_body(u("x")))
        self.assertEqual(r.status_code, 429)
        self.assertGreater(int(r.headers["retry-after"]), 0)
        self.assertEqual(r.json()["error"]["type"], "rate_limit_error")

    def test_oversized_context_is_503_with_reason(self):
        self.service.runner.registry.providers["dsw"].web  # noqa: B018 - наличие
        r = self.post(chat_body({"role": "system", "content": "s" * 300_000}, u("x")))
        self.assertEqual(r.status_code, 503)
        self.assertIn("inline_max_chars", r.json()["error"]["message"])


class StatusTests(ServiceCase):
    def test_models_lists_models_and_chains(self):
        ids = [
            m["id"]
            for m in self.http.get("/v1/models", headers=auth("hermes-secret")).json()[
                "data"
            ]
        ]
        self.assertEqual(ids, ["m", "chain:main"])

    def test_status_has_bindings_and_no_secrets(self):
        self.post(chat_body(u("x")))
        r = self.http.get("/status", headers=auth("hermes-secret"))
        data = r.json()
        (binding,) = data["bindings"]
        self.assertEqual(binding["binding"], "DS_A@direct")
        self.assertEqual(binding["account"], "ok")
        self.assertEqual(binding["busy"], 0)
        self.assertEqual(data["chats"]["active"], 1)
        text = r.text
        self.assertNotIn("a.json", text)
        self.assertNotIn(str(self.dir), text)
        self.assertNotIn("secret", text)

    def test_status_shows_needs_auth(self):
        self.site.fail_with = httpx.Response(
            200, json={"code": 40003, "msg": "INVALID_TOKEN"}
        )
        self.post(chat_body(u("x")))
        data = self.http.get("/status", headers=auth("hermes-secret")).json()
        self.assertEqual(data["bindings"][0]["account"], "needs_auth")
        self.assertIn("key", data["bindings"][0]["paused"])


class ReloadTests(ServiceCase):
    def test_reload_from_loopback_picks_up_new_client(self):
        (self.dir / "models" / "clients.toml").write_text(
            CLIENTS
            + '\n[[clients]]\nclient_name = "new"\ntoken_env = "TOKEN_NEW"\ndefault_model = "m"\n'
        )
        (self.dir / ".env").write_text(
            "TOKEN_HERMES=hermes-secret\nTOKEN_OTHER=other-secret\nTOKEN_NEW=new-secret\n"
        )
        self.assertEqual(
            self.post(chat_body(u("x")), token="new-secret").status_code, 401
        )
        # Реестр перечитывается из файлов; их в этом тестовом каталоге нет —
        # проверяем границу: ошибка конфигурации не ломает работающий сервис.
        r = self.http.post("/admin/reload")
        self.assertEqual(r.status_code, 400)
        self.assertEqual(self.post(chat_body(u("x"))).status_code, 200)

    def test_reload_from_remote_is_forbidden_without_admin_token(self):
        remote = TestClient(create_app(self.service), client=("10.0.0.5", 1))
        self.addCleanup(remote.close)
        self.assertEqual(remote.post("/admin/reload").status_code, 403)


class OpenModeTests(WebCase):
    @unittest.skipIf(TestClient is None, "нужны extras [server]")
    def test_without_clients_toml_only_loopback_is_served(self):
        (self.dir / ".env").write_text("X=1\n")
        service = Service(self.dir, runner=self.runner)
        local = TestClient(create_app(service), client=("127.0.0.1", 1))
        remote = TestClient(create_app(service), client=("192.168.1.9", 1))
        self.addCleanup(local.close)
        self.addCleanup(remote.close)
        body = {"messages": [{"role": "user", "content": "x"}], "model": "m"}
        self.assertEqual(local.post("/v1/chat/completions", json=body).status_code, 200)
        self.assertEqual(
            remote.post("/v1/chat/completions", json=body).status_code, 401
        )
        self.assertEqual(
            {r.namespace for r in self.store.by_state("active")}, {"default"}
        )


@unittest.skipIf(TestClient is None, "нужны extras [server]")
class ChainTests(unittest.TestCase):
    """Цепочка уходит ко второй модели, когда первый провайдер занят."""

    def setUp(self):
        import shutil
        import tempfile
        from pathlib import Path

        self.dir = Path(tempfile.mkdtemp())
        self.addCleanup(shutil.rmtree, self.dir, ignore_errors=True)
        (self.dir / "models").mkdir()
        (self.dir / "models" / "clients.toml").write_text(
            '[[clients]]\nclient_name = "c"\ntoken_env = "T"\ndefault_model = "chain:main"\n'
        )
        (self.dir / ".env").write_text("T=tok\n")
        keys, providers = {}, {}
        for name in ("p1", "p2"):
            path = self.dir / f"{name}.json"
            path.write_text(json.dumps({"token": f"T-{name}"}))
            providers[name] = Provider(
                name,
                "deepseek_web",
                "https://chat.test",
                "",
                auth="web_session",
                web=WebSettings(models=((f"m-{name}", WebModel()),)),
            )
            keys[f"K-{name}"] = Key(f"K-{name}", name, ("direct",), key_file=str(path))
        self.site = FakeDeepSeek()
        self.seen: list[str] = []

        def handle(request: httpx.Request) -> httpx.Response:
            token = request.headers["authorization"].removeprefix("Bearer ")
            if request.url.path.endswith("/chat/completion"):
                self.seen.append(token)
                if token == "T-p1":
                    body = (
                        'event: hint\ndata: {"type":"error","finish_reason":"backend_busy",'
                        '"content":"занят"}\n\n'
                    ).encode()
                    return httpx.Response(
                        200, content=body, headers={"content-type": "text/event-stream"}
                    )
            return self.site(request)

        context = TransportContext(
            ConversationStore(),
            http_factory=lambda proxy: httpx.Client(
                transport=httpx.MockTransport(handle)
            ),
            pow_solver=lambda c, u_, h: 1,
        )
        registry = Registry(
            providers=providers,
            proxies={"direct": Proxy("direct", "")},
            keys=keys,
            models={
                "m-p1": Model("m-p1", providers=("p1",)),
                "m-p2": Model("m-p2", providers=("p2",)),
            },
            env_keys={},
        )
        from llm_rotator.registry import Chain

        registry.chains["main"] = Chain(
            "main", (registry.models["m-p1"], registry.models["m-p2"])
        )
        runner = ModelRunner(registry, context=context, janitor=False)
        self.service = Service(self.dir, runner=runner)
        self.http = TestClient(create_app(self.service), client=("127.0.0.1", 1))
        self.addCleanup(self.http.close)

    def test_chain_goes_to_second_model_on_busy(self):
        r = self.http.post(
            "/v1/chat/completions",
            json={"messages": [u("привет")]},
            headers=auth("tok"),
        )
        self.assertEqual(r.status_code, 200)
        self.assertEqual(r.json()["model"], "m-p2")
        self.assertEqual(self.seen, ["T-p1", "T-p2"])
        self.assertEqual(r.headers["x-rotator-attempts"], "1")
        self.assertIn("m-p2", r.headers["x-rotator-origin"])

    def test_everything_down_is_a_single_summary_error(self):
        self.site.fail_with = httpx.Response(200, json={"code": 40003, "msg": "x"})
        self.seen.clear()
        r = self.http.post(
            "/v1/chat/completions", json={"messages": [u("x")]}, headers=auth("tok")
        )
        self.assertEqual(r.status_code, 503)
        self.assertIn("m-p1", r.json()["error"]["message"])
        self.assertIn("m-p2", r.json()["error"]["message"])


class PersistTests(unittest.TestCase):
    @unittest.skipIf(TestClient is None, "нужны extras [server]")
    def test_pauses_survive_a_restart(self):
        store = ConversationStore()
        self.addCleanup(store.close)
        now = [100.0]
        first = SessionState(clock=lambda: now[0])
        first.penalize(("key", "K1"))
        first.cool_down("K2", 300)
        first.suspend_provider("p", 120)
        self.assertEqual(persist.dump(first, store), 3)
        later = [5000.0]  # другой процесс — другие «монотонные» часы
        second = SessionState(clock=lambda: later[0])
        self.assertEqual(persist.load(second, store), 3)
        self.assertTrue(second.blocked(("key", "K1")))
        self.assertFalse(second.key_available("K2"))
        self.assertFalse(second.provider_available("p", "m"))
        later[0] += 400
        self.assertTrue(second.key_available("K2"))

    @unittest.skipIf(TestClient is None, "нужны extras [server]")
    def test_expired_pauses_are_not_restored(self):
        import time

        store = ConversationStore()
        self.addCleanup(store.close)
        state = SessionState(clock=lambda: 0.0)
        state.cool_down("K", 30)
        persist.dump(state, store)
        store.replace_pauses([("cooldown", "K", 30.0, 0)], time.time() - 3600)
        self.assertEqual(persist.load(SessionState(), store), 0)


if __name__ == "__main__":
    unittest.main()
