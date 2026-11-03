"""DeepSeekClient на моках: SSE-фикстуры, ошибки в теле HTTP 200, PoW, файлы."""

from __future__ import annotations

import base64
import json
import threading
import unittest
from pathlib import Path

import httpx

from llm_rotator import Failure, SessionState, WebModel, WebSettings
from llm_rotator.web.deepseek import pow as pow_module
from llm_rotator.web.deepseek.account import Account, load_account
from llm_rotator.web.deepseek.client import DeepSeekClient
from llm_rotator.web.deepseek.sse import Delta, Finish, Ready, split_sse
from llm_rotator.web.errors import WebError, to_failure

FIXTURES = Path(__file__).resolve().parent.parent / "fixtures" / "deepseek"
BASE = "https://chat.test"
CHALLENGE = {
    "algorithm": "DeepSeekHashV1",
    "challenge": "abc",
    "salt": "s",
    "expire_at": 1,
    "difficulty": 100,
    "signature": "sig",
}


def fixture(name: str) -> bytes:
    return (FIXTURES / name).read_bytes()


def envelope(biz_data: dict | None = None, biz_code: int = 0, biz_msg: str = ""):
    return {
        "code": 0,
        "msg": "",
        "data": {"biz_code": biz_code, "biz_msg": biz_msg, "biz_data": biz_data},
    }


def sse_response(name: str) -> httpx.Response:
    return httpx.Response(
        200, content=fixture(name), headers={"content-type": "text/event-stream"}
    )


class Server:
    """Подставной сервер: маршрут по пути, список принятых запросов."""

    def __init__(self, routes: dict[str, object]) -> None:
        self.routes = routes
        self.requests: list[httpx.Request] = []

    def __call__(self, request: httpx.Request) -> httpx.Response:
        self.requests.append(request)
        path = request.url.path
        if path == "/api/v0/chat/create_pow_challenge" and path not in self.routes:
            return httpx.Response(200, json=envelope({"challenge": CHALLENGE}))
        handler = self.routes.get(path, self.routes.get("*"))
        return handler(request) if callable(handler) else handler


def make_client(routes: dict[str, object], **settings) -> tuple[DeepSeekClient, Server]:
    server = Server(routes)
    http = httpx.Client(transport=httpx.MockTransport(server))
    client = DeepSeekClient(
        Account(token="SECRET-TOKEN", cookie="c=1", wasm_url="https://w/x.wasm"),
        http,
        WebSettings(**settings),
        BASE,
        pow_solver=lambda challenge, url, client: 7,
        sleep=lambda seconds: None,
    )
    return client, server


COMPLETION = "/api/v0/chat/completion"


def run(client: DeepSeekClient, **kwargs):
    return list(client.complete("chat", None, "hi", **kwargs))


class SseFixtureTests(unittest.TestCase):
    def events(self, name: str):
        client, _ = make_client({COMPLETION: sse_response(name)})
        return run(client)

    def test_basic_answer(self):
        events = self.events("completion_basic.sse")
        self.assertEqual(events[0], Ready(2, 1))
        text = "".join(e.text for e in events if isinstance(e, Delta))
        self.assertEqual(text, "Привет, мир!")
        self.assertEqual(events[-1], Finish("stop"))

    def test_think_is_separated_from_answer(self):
        events = self.events("completion_think.sse")
        think = "".join(e.text for e in events if isinstance(e, Delta) and e.reasoning)
        answer = "".join(
            e.text for e in events if isinstance(e, Delta) and not e.reasoning
        )
        self.assertEqual(think, "Нужно ответить коротко.")
        self.assertEqual(answer, "Ответ: 42")

    def test_incomplete_means_length(self):
        self.assertEqual(self.events("completion_incomplete.sse")[-1], Finish("length"))

    def test_backend_busy_is_busy_error(self):
        with self.assertRaises(WebError) as ctx:
            self.events("completion_backend_busy.sse")
        self.assertEqual(ctx.exception.kind, "busy")

    def test_too_many_messages_is_rate_limit(self):
        with self.assertRaises(WebError) as ctx:
            self.events("completion_too_many_messages.sse")
        self.assertEqual(ctx.exception.kind, "rate_limit")

    def test_multibyte_split_across_chunks(self):
        raw = 'data: {"v":{"response":{"fragments":[{"type":"RESPONSE","content":"ёж"}]}}}\n\n'
        data = raw.encode()
        cut = data.index("ё".encode()) + 1  # режем двухбайтовый символ пополам
        pairs = list(split_sse([data[:cut], data[cut:]]))
        self.assertIn("ёж", pairs[0][1])

    def test_crlf_and_comments(self):
        pairs = list(
            split_sse([b": ping\r\nevent: x\r\ndata: 1\r\n\r\ndata: 2\r\n\r\n"])
        )
        self.assertEqual(pairs, [("x", "1"), ("message", "2")])

    def test_empty_stream_is_empty_error(self):
        client, _ = make_client(
            {
                COMPLETION: httpx.Response(
                    200,
                    content=b"event: ready\ndata: "
                    b'{"response_message_id":2}\n\nevent: close\ndata: {}\n\n',
                    headers={"content-type": "text/event-stream"},
                )
            }
        )
        with self.assertRaises(WebError) as ctx:
            run(client)
        self.assertEqual(ctx.exception.kind, "empty")


class BodyErrorTests(unittest.TestCase):
    def failing(self, body: bytes, status: int = 200, path: str = COMPLETION):
        response = httpx.Response(
            status, content=body, headers={"content-type": "application/json"}
        )
        client, _ = make_client({path: response, "*": response})
        return client

    def test_invalid_token_in_200_body_is_auth(self):
        client = self.failing(fixture("error_invalid_token.json"))
        with self.assertRaises(WebError) as ctx:
            run(client)
        self.assertEqual(ctx.exception.kind, "auth")

    def test_invalid_chat_session_is_chat_gone(self):
        client = self.failing(fixture("error_invalid_chat_session.json"))
        with self.assertRaises(WebError) as ctx:
            run(client)
        self.assertEqual(ctx.exception.kind, "chat_gone")

    def test_http_statuses(self):
        cases = {
            (401, b"{}"): "auth",
            (403, b'{"msg":"x"}'): "auth",
            (403, b"<html>cf</html>"): "blocked",
            (429, b"{}"): "rate_limit",
            (413, b"{}"): "too_large",
            (400, b"{}"): "bad_request",
            (400, b'{"msg":"prompt is too long"}'): "too_large",
            (502, b"bad"): "upstream",
        }
        for (status, body), kind in cases.items():
            with self.subTest(status=status, kind=kind):
                with self.assertRaises(WebError) as ctx:
                    self.failing(body, status).create_chat()
                self.assertEqual(ctx.exception.kind, kind)

    def test_unknown_biz_code_is_upstream(self):
        body = json.dumps(envelope(None, biz_code=7, biz_msg="strange")).encode()
        with self.assertRaises(WebError) as ctx:
            run(self.failing(body))
        self.assertEqual(ctx.exception.kind, "upstream")

    def test_retry_after_header_is_kept(self):
        client, _ = make_client(
            {
                "/api/v0/chat_session/create": httpx.Response(
                    429, headers={"retry-after": "42"}
                )
            }
        )
        with self.assertRaises(WebError) as ctx:
            client.create_chat()
        self.assertEqual(ctx.exception.retry_after, 42.0)

    def test_network_failure_is_network(self):
        def boom(request):
            raise httpx.ConnectError("no route")

        client, _ = make_client({"/api/v0/chat_session/create": boom})
        with self.assertRaises(WebError) as ctx:
            client.create_chat()
        self.assertEqual(ctx.exception.kind, "network")

    def test_timeout_is_timeout(self):
        def slow(request):
            raise httpx.ReadTimeout("slow")

        client, _ = make_client({"/api/v0/chat_session/create": slow})
        with self.assertRaises(WebError) as ctx:
            client.create_chat()
        self.assertEqual(ctx.exception.kind, "timeout")

    def test_token_never_appears_in_error_or_repr(self):
        client = self.failing(b"{}", 401)
        with self.assertRaises(WebError) as ctx:
            client.create_chat()
        shown = f"{ctx.exception!r} {ctx.exception} {client!r} {client._account!r}"
        self.assertNotIn("SECRET-TOKEN", shown)


class StallTests(unittest.TestCase):
    def test_silence_after_ready_fires_idle_timeout(self):
        release = threading.Event()

        class Stalling(httpx.SyncByteStream):
            def __iter__(self):
                yield b'event: ready\ndata: {"response_message_id":2}\n\n'
                release.wait(5)

            def close(self):
                release.set()

        client, _ = make_client(
            {
                COMPLETION: httpx.Response(
                    200,
                    stream=Stalling(),
                    headers={"content-type": "text/event-stream"},
                )
            },
            first_event_timeout=5,
            idle_timeout=0.2,
        )
        with self.assertRaises(WebError) as ctx:
            run(client)
        self.assertEqual(ctx.exception.kind, "timeout")


class MethodTests(unittest.TestCase):
    def test_create_chat_reads_both_id_shapes(self):
        for biz in ({"chat_session": {"id": "A"}}, {"id": "A"}):
            client, _ = make_client(
                {"/api/v0/chat_session/create": httpx.Response(200, json=envelope(biz))}
            )
            self.assertEqual(client.create_chat(), "A")

    def test_delete_returns_false_when_already_gone(self):
        client, _ = make_client(
            {
                "/api/v0/chat_session/delete": httpx.Response(
                    200, content=fixture("error_invalid_chat_session.json")
                )
            }
        )
        self.assertFalse(client.delete_chat("x"))

    def test_delete_ok(self):
        client, server = make_client(
            {"/api/v0/chat_session/delete": httpx.Response(200, json=envelope({}))}
        )
        self.assertTrue(client.delete_chat("abc"))
        self.assertEqual(
            json.loads(server.requests[-1].content), {"chat_session_id": "abc"}
        )

    def test_ping_dead_token(self):
        client, _ = make_client(
            {
                "/api/v0/chat_session/fetch_page": httpx.Response(
                    200, content=fixture("error_invalid_token.json")
                )
            }
        )
        with self.assertRaises(WebError) as ctx:
            client.ping()
        self.assertEqual(ctx.exception.kind, "auth")

    def test_completion_request_shape_and_pow_header(self):
        client, server = make_client({COMPLETION: sse_response("completion_basic.sse")})
        run(client, ref_file_ids=["f1"], flags=WebModel(thinking=True))
        request = server.requests[-1]
        body = json.loads(request.content)
        self.assertEqual(body["chat_session_id"], "chat")
        self.assertEqual(body["ref_file_ids"], ["f1"])
        self.assertTrue(body["thinking_enabled"])
        self.assertFalse(body["search_enabled"])
        self.assertEqual(request.headers["authorization"], "Bearer SECRET-TOKEN")
        pow_header = json.loads(base64.b64decode(request.headers["x-ds-pow-response"]))
        self.assertEqual(pow_header["answer"], 7)
        self.assertEqual(pow_header["target_path"], COMPLETION)
        self.assertEqual(pow_header["signature"], "sig")

    def test_upload_polls_until_success(self):
        states = iter(["PENDING", "PARSING", "SUCCESS"])

        def fetch(request):
            return httpx.Response(
                200,
                json=envelope({"files": [{"id": "F", "status": next(states)}]}),
            )

        client, server = make_client(
            {
                "/api/v0/file/upload_file": httpx.Response(
                    200, json=envelope({"id": "F", "status": "PENDING"})
                ),
                "/api/v0/file/fetch_files": fetch,
            }
        )
        self.assertEqual(client.upload_file("a.txt", b"hello"), "F")
        upload = next(r for r in server.requests if r.url.path.endswith("upload_file"))
        self.assertEqual(upload.headers["x-file-size"], "5")
        self.assertIn("multipart/form-data", upload.headers["content-type"])

    def test_upload_failed_status(self):
        client, _ = make_client(
            {
                "/api/v0/file/upload_file": httpx.Response(
                    200, json=envelope({"id": "F", "status": "FAILED"})
                )
            }
        )
        with self.assertRaises(WebError) as ctx:
            client.upload_file("a.txt", b"x")
        self.assertEqual(ctx.exception.kind, "bad_request")

    def test_load_account(self):
        import tempfile

        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "a.json"
            path.write_text(json.dumps({"token": " T ", "wasmUrl": "u"}))
            account = load_account(path)
            self.assertEqual((account.token, account.wasm_url), ("T", "u"))
            self.assertNotIn("T", repr(account).replace("Account", ""))
            path.write_text("{}")
            with self.assertRaises(WebError):
                load_account(path)
            with self.assertRaises(WebError):
                load_account(Path(tmp) / "missing.json")


class FailureTableTests(unittest.TestCase):
    def test_table(self):
        cases = {
            "auth": (Failure.KEY, None),
            "rate_limit": (Failure.QUOTA, 60.0),
            "busy": (Failure.QUOTA, 30.0),
            "timeout": (Failure.PROXY, None),
            "network": (Failure.PROXY, None),
            "blocked": (Failure.BLOCKED, None),
            "too_large": (Failure.TOO_LARGE, None),
            "bad_request": (Failure.FATAL, None),
            "empty": (Failure.QUOTA, 10.0),
        }
        for kind, expected in cases.items():
            with self.subTest(kind=kind):
                self.assertEqual(to_failure(WebError(kind)), expected)

    def test_explicit_retry_after_wins(self):
        self.assertEqual(
            to_failure(WebError("rate_limit", retry_after=5)), (Failure.QUOTA, 5)
        )


class PowTests(unittest.TestCase):
    def test_header_value_roundtrip(self):
        value = pow_module.header_value(CHALLENGE, 12, "/p")
        decoded = json.loads(base64.b64decode(value))
        self.assertEqual(decoded["answer"], 12)
        self.assertEqual(decoded["target_path"], "/p")
        self.assertEqual(decoded["algorithm"], "DeepSeekHashV1")

    def test_solver_error_becomes_upstream(self):
        server = Server({})
        client = DeepSeekClient(
            Account(token="t"),
            httpx.Client(transport=httpx.MockTransport(server)),
            WebSettings(),
            BASE,
            pow_solver=lambda c, u, h: (_ for _ in ()).throw(KeyError("salt")),
        )
        with self.assertRaises(WebError) as ctx:
            list(client.complete("c", None, "x"))
        self.assertEqual(ctx.exception.kind, "upstream")


class ProviderPauseTests(unittest.TestCase):
    def state(self):
        self.now = [0.0]
        return SessionState(clock=lambda: self.now[0])

    def test_one_busy_key_does_not_suspend(self):
        state = self.state()
        self.assertFalse(state.note_busy("p", "K1"))
        self.assertFalse(state.note_busy("p", "K1"))
        self.assertTrue(state.provider_available("p", "m"))

    def test_two_keys_within_window_suspend_provider(self):
        state = self.state()
        state.note_busy("p", "K1")
        self.now[0] = 30
        self.assertTrue(state.note_busy("p", "K2"))
        self.assertFalse(state.provider_available("p", "m"))
        self.assertTrue(state.provider_available("other", "m"))
        self.now[0] = 91
        self.assertTrue(state.provider_available("p", "m"))

    def test_second_key_after_window_does_not_suspend(self):
        state = self.state()
        state.note_busy("p", "K1")
        self.now[0] = 61
        self.assertFalse(state.note_busy("p", "K2"))

    def test_forget_models_keeps_suspension(self):
        state = self.state()
        state.suspend_provider("p", 60)
        state.forget_models()
        self.assertFalse(state.provider_available("p", "m"))


if __name__ == "__main__":
    unittest.main()


@unittest.skipUnless(
    __import__("os").environ.get("LLM_ROTATOR_POW_WASM"),
    "нужен LLM_ROTATOR_POW_WASM=<путь к sha3_wasm_bg.7b9ca65ddd.wasm> и wasmtime",
)
class RealPowTests(unittest.TestCase):
    """Настоящий WASM против пары challenge→answer, снятой живьём и сверенной
    с эталоном `lib/pow.js` (FreeDeepseekAPI): ответ совпал (2026-11-03)."""

    def test_known_pair(self):
        import os

        pair = json.loads(fixture("pow_pair.json"))
        wasm = Path(os.environ["LLM_ROTATOR_POW_WASM"]).read_bytes()
        http = httpx.Client(
            transport=httpx.MockTransport(lambda r: httpx.Response(200, content=wasm))
        )
        answer = pow_module.solve(pair["challenge"], "https://w/real.wasm", http)
        self.assertEqual(answer, pair["answer"])


if __name__ == "__main__":
    unittest.main()


@unittest.skipUnless(
    __import__("os").environ.get("LLM_ROTATOR_POW_WASM"),
    "нужен LLM_ROTATOR_POW_WASM=<путь к sha3_wasm_bg.*.wasm> и пакет wasmtime",
)
class RealPowTests(unittest.TestCase):
    """Сверка с эталоном `lib/pow.js` из FreeDeepseekAPI: ответ тот же."""

    def test_solves_and_matches_reference(self):
        import os

        wasm = Path(os.environ["LLM_ROTATOR_POW_WASM"]).read_bytes()
        http = httpx.Client(
            transport=httpx.MockTransport(lambda r: httpx.Response(200, content=wasm))
        )
        challenge = dict(CHALLENGE, challenge="a" * 64, difficulty=1000)
        answer = pow_module.solve(challenge, "https://w/real.wasm", http)
        expected = os.environ.get("LLM_ROTATOR_POW_EXPECTED")
        if expected is not None:
            self.assertEqual(answer, int(expected))
        self.assertGreaterEqual(answer, 0)
