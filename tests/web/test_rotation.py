"""R3: порядок связок и быстрый уход с недоступного аккаунта."""

from __future__ import annotations

import json
import shutil
import tempfile
import threading
import time
import unittest
from pathlib import Path

import httpx

from llm_rotator import (
    ConversationHint,
    Failure,
    GatewayError,
    Key,
    Model,
    ModelRunner,
    Provider,
    Proxy,
    Registry,
    WebModel,
    WebSettings,
)
from llm_rotator.web.context import TransportContext
from llm_rotator.web.store import ConversationStore

from .test_transport import FakeDeepSeek, assistant, user


class TwoAccounts(unittest.TestCase):
    first_event_timeout = 15
    proxies = ("p1", "p2")
    max_concurrent = 1

    def setUp(self):
        self.dir = Path(tempfile.mkdtemp())
        self.addCleanup(shutil.rmtree, self.dir, ignore_errors=True)
        self.site = FakeDeepSeek()
        self.used: list[str] = []  # токены, которыми шли completion
        self.behavior: dict[str, str] = {}  # токен -> "stall" | "dead"
        self.release = threading.Event()
        self.addCleanup(self.release.set)
        keys = {}
        for name, proxy in zip(("DS_A", "DS_B"), self.proxies):
            path = self.dir / f"{name}.json"
            path.write_text(json.dumps({"token": f"T-{name}", "wasmUrl": "u"}))
            keys[name] = Key(
                name,
                "dsw",
                (proxy,),
                key_file=str(path),
                max_concurrent=self.max_concurrent,
            )
        self.store = ConversationStore()
        self.addCleanup(self.store.close)
        self.context = TransportContext(
            self.store,
            http_factory=lambda proxy: httpx.Client(
                transport=httpx.MockTransport(self.handle)
            ),
            pow_solver=lambda c, u, h: 1,
        )
        web = WebSettings(
            models=(("m", WebModel()),), first_event_timeout=self.first_event_timeout
        )
        provider = Provider(
            "dsw", "deepseek_web", "https://chat.test", "", auth="web_session", web=web
        )
        proxies = {
            n: Proxy(n, "" if n == "direct" else f"http://{n}:1") for n in self.proxies
        }
        registry = Registry(
            providers={"dsw": provider},
            proxies=proxies,
            keys=keys,
            models={"m": Model("m", providers=("dsw",))},
            env_keys={},
        )
        self.runner = ModelRunner(registry, context=self.context, janitor=False)
        self.addCleanup(self.runner.close)

    def handle(self, request: httpx.Request) -> httpx.Response:
        token = request.headers["authorization"].removeprefix("Bearer ")
        if request.url.path == "/api/v0/chat/completion":
            self.used.append(token)
            mode = self.behavior.get(token)
            if mode == "dead":
                return httpx.Response(200, json={"code": 40003, "msg": "INVALID_TOKEN"})
            if mode == "stall":
                release = self.release

                class Silent(httpx.SyncByteStream):
                    def __iter__(self):
                        release.wait(10)
                        return iter(())

                    def close(self):
                        release.set()

                return httpx.Response(
                    200, stream=Silent(), headers={"content-type": "text/event-stream"}
                )
        return self.site(request)

    def ask(self, messages, session=""):
        return self.runner.complete(
            "m", messages, conversation=ConversationHint("ns", session)
        )


class OrderTests(TwoAccounts):
    def test_independent_conversations_spread_by_lru(self):
        for text in ("1", "2", "3", "4"):
            self.ask([user(text)])
        self.assertEqual(self.used, ["T-DS_A", "T-DS_B", "T-DS_A", "T-DS_B"])

    def test_continuation_returns_to_the_account_that_has_the_chat(self):
        first = self.ask([user("1")])  # A
        self.ask([user("другой")])  # B — теперь A «давнее»... нет, B свежее
        self.ask([user("ещё")])  # A
        # Разговор 1 живёт на A, но по LRU первым был бы B (A использовали позже).
        self.used.clear()
        self.ask([user("1"), assistant(first.text), user("2")])
        self.assertEqual(self.used, ["T-DS_A"])
        self.assertEqual(self.site.completions[-1]["prompt"], "User: 2")

    def test_busy_account_is_tried_last_without_penalty(self):
        self.assertTrue(self.context.busy.acquire("DS_A", 1))
        self.ask([user("1")])
        self.assertEqual(self.used, ["T-DS_B"])
        state = self.runner.gateway.state
        self.assertFalse(state.cooldown_keys)
        self.context.busy.release("DS_A")
        self.ask([user("2")])
        self.assertIn("T-DS_A", self.used)

    def test_all_busy_fails_without_pausing_anyone(self):
        self.context.busy.acquire("DS_A", 1)
        self.context.busy.acquire("DS_B", 1)
        with self.assertRaises(GatewayError) as ctx:
            self.ask([user("1")])
        self.assertTrue(all(a.transient for a in ctx.exception.attempts))
        self.assertFalse(self.runner.gateway.state.cooldown_keys)
        self.context.busy.release("DS_A")
        self.assertTrue(self.ask([user("1")]).text)


class StallTests(TwoAccounts):
    first_event_timeout = 0.3

    def test_silent_account_hands_over_within_first_event_timeout(self):
        self.behavior["T-DS_A"] = "stall"
        started = time.monotonic()
        response = self.ask([user("1")])
        elapsed = time.monotonic() - started
        self.assertTrue(response.text)
        self.assertEqual(self.used, ["T-DS_A", "T-DS_B"])
        self.assertLess(elapsed, 0.3 + 0.7)
        self.assertEqual(response.origin.key_name, "DS_B")
        first = response.attempts[0]
        self.assertEqual((first.key_name, first.failure), ("DS_A", Failure.PROXY))
        # Чат, начатый на молчавшем аккаунте, ушёл в брошенные.
        self.assertEqual(
            {r.key_name for r in self.store.by_state("abandoned")}, {"DS_A"}
        )

    def test_second_request_skips_the_silent_account_for_a_while(self):
        self.behavior["T-DS_A"] = "stall"
        self.ask([user("1")])
        self.used.clear()
        self.ask([user("2")])
        self.assertEqual(self.used, ["T-DS_B"])


class StallOnDirectTests(StallTests):
    """Пока прокси не закреплены, у всех аккаунтов `direct`: молчание одного
    не должно отрезать остальных."""

    proxies = ("direct", "direct")

    def test_silent_account_hands_over_within_first_event_timeout(self):
        self.behavior["T-DS_A"] = "stall"
        response = self.ask([user("1")])
        self.assertEqual(response.origin.key_name, "DS_B")
        # Вина на аккаунте (пауза ключа), а не на общем прямом адресе.
        self.assertEqual(response.attempts[0].failure, Failure.QUOTA)
        self.assertFalse(self.runner.gateway.state.penalties)


class DeadTokenTests(TwoAccounts):
    def test_dead_token_pauses_account_and_second_request_does_not_touch_it(self):
        self.behavior["T-DS_A"] = "dead"
        first = self.ask([user("1")])
        self.assertEqual(first.origin.key_name, "DS_B")
        self.assertEqual(first.attempts[0].failure, Failure.KEY)
        self.assertEqual(self.store.account_health("DS_A"), "needs_auth")
        self.used.clear()
        self.ask([user("2")])
        self.assertEqual(self.used, ["T-DS_B"])
        self.assertEqual(self.store.by_state("active")[0].key_name, "DS_B")


if __name__ == "__main__":
    unittest.main()
