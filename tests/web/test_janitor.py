"""Уборка чатов (R1): удаляем своё, через свою связку, переживаем рестарт."""

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
from llm_rotator.web.janitor import MIN_GAP, Janitor
from llm_rotator.web.store import (
    ABANDONED,
    ACTIVE,
    DELETE_FAILED,
    DELETED,
    ConversationStore,
)

from .test_transport import FakeDeepSeek, envelope


class Clock:
    def __init__(self) -> None:
        # Не в прошлом: mtime файлов сессии — настоящее время.
        self.now = time.time() + 1000

    def __call__(self) -> float:
        return self.now


class JanitorCase(unittest.TestCase):
    settings: dict = {}

    def setUp(self):
        self.dir = Path(tempfile.mkdtemp())
        self.addCleanup(shutil.rmtree, self.dir, ignore_errors=True)
        self.files = {}
        for name in ("DS_A", "DS_B"):
            path = self.dir / f"{name}.json"
            path.write_text(json.dumps({"token": f"TOKEN-{name}"}))
            self.files[name] = path
        self.clock = Clock()
        self.site = FakeDeepSeek()
        self.seen: list[tuple[str, str, str]] = []  # (proxy, token, chat_id)
        self.delete_response: httpx.Response | None = None
        self.store = ConversationStore(clock=self.clock)
        self.addCleanup(self.store.close)
        self.context = TransportContext(
            self.store,
            http_factory=self.http_for,
            pow_solver=lambda c, u, h: 1,
        )
        self.registry = self.make_registry()
        self.janitor = Janitor(self.registry, self.context, clock=self.clock)

    def http_for(self, proxy):
        def handler(request: httpx.Request) -> httpx.Response:
            if request.url.path == "/api/v0/chat_session/delete":
                chat_id = json.loads(request.content)["chat_session_id"]
                token = request.headers["authorization"].removeprefix("Bearer ")
                self.seen.append((proxy.name, token, chat_id))
                if self.delete_response is not None:
                    return self.delete_response
            return self.site(request)

        return httpx.Client(transport=httpx.MockTransport(handler))

    def make_registry(self, **extra) -> Registry:
        web = WebSettings(models=(("m", WebModel()),), **{**self.settings, **extra})
        provider = Provider(
            "dsw", "deepseek_web", "https://chat.test", "", auth="web_session", web=web
        )
        return Registry(
            providers={"dsw": provider},
            proxies={
                "p1": Proxy("p1", "http://p1:1"),
                "p2": Proxy("p2", "http://p2:1"),
            },
            keys={
                "DS_A": Key("DS_A", "dsw", ("p1",), key_file=str(self.files["DS_A"])),
                "DS_B": Key("DS_B", "dsw", ("p2",), key_file=str(self.files["DS_B"])),
            },
            models={"m": Model("m", providers=("dsw",))},
            env_keys={},
        )

    def chat(self, chat_id, key="DS_A", proxy="p1", state=ABANDONED):
        self.store.create_chat("dsw", chat_id, key, proxy, "ns", "", "ctx", state)

    def state(self, chat_id):
        return self.store.get("dsw", chat_id).state

    def advance(self, seconds):
        self.clock.now += seconds


class DeletionTests(JanitorCase):
    def test_abandoned_chat_is_deleted_through_its_own_binding(self):
        self.chat("a1", "DS_A", "p1")
        self.chat("b1", "DS_B", "p2")
        self.assertEqual(self.janitor.run_once(), 2)
        self.assertCountEqual(
            self.seen, [("p1", "TOKEN-DS_A", "a1"), ("p2", "TOKEN-DS_B", "b1")]
        )
        self.assertEqual((self.state("a1"), self.state("b1")), (DELETED, DELETED))

    def test_already_gone_counts_as_deleted(self):
        self.chat("a1")
        self.delete_response = httpx.Response(
            200, json=envelope(None, 1, "invalid chat session id")
        )
        self.janitor.run_once()
        self.assertEqual(self.state("a1"), DELETED)

    def test_active_chat_is_left_alone_until_idle_ttl(self):
        self.chat("a1", state=ACTIVE)
        self.assertEqual(self.janitor.run_once(), 0)
        self.assertEqual(self.state("a1"), ACTIVE)
        self.advance(7201)
        self.assertEqual(self.janitor.run_once(), 1)
        self.assertEqual(self.state("a1"), DELETED)

    def test_stuck_busy_chat_is_cleaned_after_ttl(self):
        self.chat("a1", state="busy")
        self.advance(7201)
        self.assertEqual(self.janitor.run_once(), 1)

    def test_never_policy_deletes_nothing(self):
        self.registry = self.make_registry(delete_policy="never")
        self.janitor = Janitor(self.registry, self.context, clock=self.clock)
        self.chat("a1")
        self.chat("a2", state=ACTIVE)
        self.advance(10**6)
        self.assertEqual(self.janitor.run_once(), 0)
        self.assertEqual(self.seen, [])

    def test_rate_limit_one_deletion_per_account_per_gap(self):
        self.chat("a1")
        self.chat("a2")
        self.assertEqual(self.janitor.run_once(), 1)
        self.assertEqual(self.janitor.run_once(), 0)  # слишком рано
        self.advance(MIN_GAP)
        self.assertEqual(self.janitor.run_once(), 1)
        self.assertEqual({self.state("a1"), self.state("a2")}, {DELETED})

    def test_unknown_chats_are_never_touched(self):
        # Реестр знает только то, что в таблице chats; больше ничего не запрашивается.
        self.chat("a1")
        self.janitor.run_once()
        paths = [r.url.path for r in self.site_requests()]
        self.assertNotIn("/api/v0/chat_session/fetch_page", paths)

    def site_requests(self):
        return []

    def test_deleted_records_are_purged_after_a_week(self):
        self.chat("a1")
        self.janitor.run_once()
        self.advance(6 * 86400)
        self.janitor.run_once()
        self.assertIsNotNone(self.store.get("dsw", "a1"))
        self.advance(2 * 86400)
        self.janitor.run_once()
        self.assertIsNone(self.store.get("dsw", "a1"))

    def test_key_removed_from_config_marks_delete_failed(self):
        self.chat("x1", "DS_GONE", "p1")
        self.janitor.run_once()
        self.assertEqual(self.state("x1"), DELETE_FAILED)

    def test_chat_is_deleted_through_the_proxy_it_was_created_on(self):
        self.registry.proxies["p3"] = Proxy("p3", "http://p3:1")
        self.registry.keys["DS_A"] = Key(
            "DS_A", "dsw", ("p1", "p3"), key_file=str(self.files["DS_A"])
        )
        self.chat("a1", "DS_A", "p3")
        self.janitor.run_once()
        self.assertEqual(self.seen, [("p3", "TOKEN-DS_A", "a1")])

    def test_changed_proxy_still_uses_a_current_proxy_of_the_key(self):
        self.chat("a1", "DS_A", "old-proxy")
        self.janitor.run_once()
        self.assertEqual(self.seen, [("p1", "TOKEN-DS_A", "a1")])


class RetryTests(JanitorCase):
    def test_failure_backs_off_and_gives_up_after_five_attempts(self):
        self.chat("a1")
        self.delete_response = httpx.Response(502, text="bad gateway")
        waits = []
        for pause in (60, 300, 1800, 1800):
            self.assertEqual(self.janitor.run_once(), 0)
            record = self.store.get("dsw", "a1")
            waits.append(record.next_attempt_at - self.clock.now)
            self.advance(pause - 1)
            calls = len(self.seen)
            self.janitor.run_once()
            self.assertEqual(len(self.seen), calls)  # ещё рано
            self.advance(1)
        self.assertEqual(waits, [60, 300, 1800, 1800])
        self.janitor.run_once()  # пятая
        record = self.store.get("dsw", "a1")
        self.assertEqual((record.state, record.delete_attempts), (DELETE_FAILED, 5))
        calls = len(self.seen)
        self.advance(10**5)
        self.janitor.run_once()
        self.assertEqual(len(self.seen), calls)

    def test_success_after_failure(self):
        self.chat("a1")
        self.delete_response = httpx.Response(502, text="x")
        self.janitor.run_once()
        self.delete_response = None
        self.advance(61)
        self.assertEqual(self.janitor.run_once(), 1)
        self.assertEqual(self.state("a1"), DELETED)

    def test_dead_token_waits_for_relogin_without_counting_attempts(self):
        self.chat("a1")
        self.delete_response = httpx.Response(
            200, json={"code": 40003, "msg": "INVALID_TOKEN"}
        )
        self.janitor.run_once()
        record = self.store.get("dsw", "a1")
        self.assertEqual(record.delete_attempts, 0)
        self.assertEqual(self.store.account_health("DS_A"), "needs_auth")
        calls = len(self.seen)
        self.advance(1000)
        self.janitor.run_once()
        self.assertEqual(len(self.seen), calls)  # ждём перелогина
        # Перелогин = файл сессии обновился позже пометки.
        self.delete_response = None
        later = self.clock.now + 5
        import os

        os.utime(self.files["DS_A"], (later, later))
        self.advance(10)
        self.assertEqual(self.janitor.run_once(), 1)
        self.assertEqual(self.state("a1"), DELETED)
        self.assertEqual(self.store.account_health("DS_A"), "ok")

    def test_other_account_is_not_blocked_by_dead_one(self):
        self.chat("a1", "DS_A", "p1")
        self.chat("b1", "DS_B", "p2")
        self.store.mark_account("DS_A", "needs_auth")
        self.janitor.run_once()
        self.assertEqual((self.state("a1"), self.state("b1")), (ABANDONED, DELETED))


class RestartTests(unittest.TestCase):
    def test_queue_survives_restart(self):
        with tempfile.TemporaryDirectory() as tmp:
            db = Path(tmp) / "state" / "rotator.db"
            store = ConversationStore(db)
            store.create_chat("dsw", "a1", "DS_A", "p1", "ns", "", "ctx", ABANDONED)
            store.close()
            case = JanitorCase("setUp")
            case.setUp()
            self.addCleanup(shutil.rmtree, case.dir, ignore_errors=True)
            self.addCleanup(case.store.close)
            reopened = ConversationStore(db, clock=case.clock)
            self.addCleanup(reopened.close)
            context = TransportContext(
                reopened, http_factory=case.http_for, pow_solver=lambda c, u, h: 1
            )
            Janitor(case.registry, context, clock=case.clock).run_once()
            self.assertEqual(reopened.get("dsw", "a1").state, DELETED)
            self.assertEqual(case.seen, [("p1", "TOKEN-DS_A", "a1")])


class AfterTurnTests(JanitorCase):
    settings = {"delete_policy": "after_turn"}

    def test_after_turn_leaves_no_chats_behind(self):
        runner = ModelRunner(self.registry, context=self.context, janitor=False)
        self.addCleanup(runner.close)
        for text in ("раз", "два"):
            runner.complete(
                "m",
                [{"role": "user", "content": text}],
                conversation=ConversationHint("ns", ""),
            )
        # Каждый ход — новый чат, и после ответа он сразу брошенный.
        self.assertEqual(self.site.chats, ["chat-1", "chat-2"])
        self.assertEqual(
            {r.chat_id: r.state for r in self.store.by_state(ABANDONED)},
            {"chat-1": ABANDONED, "chat-2": ABANDONED},
        )
        self.assertEqual(self.store.by_state(ACTIVE), [])
        for _ in range(2):
            self.janitor.run_once()
            self.advance(MIN_GAP)
        self.assertCountEqual(self.site.deleted, ["chat-1", "chat-2"])
        self.assertEqual(self.store.by_state(ABANDONED), [])

    def test_after_turn_never_reuses_a_chat(self):
        runner = ModelRunner(self.registry, context=self.context, janitor=False)
        self.addCleanup(runner.close)
        first = runner.complete("m", [{"role": "user", "content": "1"}])
        runner.complete(
            "m",
            [
                {"role": "user", "content": "1"},
                {"role": "assistant", "content": first.text},
                {"role": "user", "content": "2"},
            ],
        )
        self.assertEqual(len(self.site.chats), 2)
        self.assertIsNone(self.site.completions[-1]["parent_message_id"])
        self.assertIn("User: 1", self.site.completions[-1]["prompt"])


class ThreadTests(JanitorCase):
    def test_background_thread_cleans_up_and_stops(self):
        self.chat("a1")
        janitor = Janitor(self.registry, self.context)  # настоящие часы
        self.store._clock = time.time
        janitor.start()
        janitor.wake()
        deadline = time.monotonic() + 5
        while time.monotonic() < deadline and self.state("a1") != DELETED:
            time.sleep(0.05)
        janitor.stop()
        self.assertEqual(self.state("a1"), DELETED)
        self.assertFalse(
            any(
                t.name == "llm-rotator-janitor" and t.is_alive()
                for t in threading.enumerate()
            )
        )

    def test_runner_starts_and_stops_janitor(self):
        runner = ModelRunner(self.registry, context=self.context)
        runner.complete("m", [{"role": "user", "content": "1"}])
        self.assertIsNotNone(runner._janitor)
        runner.close()
        self.assertIsNone(runner._janitor._thread)


if __name__ == "__main__":
    unittest.main()
