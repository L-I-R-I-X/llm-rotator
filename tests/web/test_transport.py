"""Разговоры на веб-чате: изоляция и переиспользование чатов (R2), на моках."""

from __future__ import annotations

import json
import shutil
import tempfile
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
from llm_rotator.transport import ChatRequest
from llm_rotator.web import fingerprint
from llm_rotator.web.busy import BusyTracker
from llm_rotator.web.context import TransportContext
from llm_rotator.web.planner import plan_turn
from llm_rotator.web.store import ABANDONED, ACTIVE, BUSY, ConversationStore

CHALLENGE = {
    "algorithm": "DeepSeekHashV1",
    "challenge": "abc",
    "salt": "s",
    "expire_at": 1,
    "difficulty": 1,
    "signature": "sig",
}


def envelope(biz_data=None, biz_code=0, biz_msg=""):
    return {
        "code": 0,
        "msg": "",
        "data": {"biz_code": biz_code, "biz_msg": biz_msg, "biz_data": biz_data},
    }


def sse(text: str, response_id: int, think: str = "") -> bytes:
    frags = []
    if think:
        frags.append({"id": 2, "type": "THINK", "content": think})
    frags.append({"id": 3, "type": "RESPONSE", "content": text})
    head = {"response_message_id": response_id, "request_message_id": response_id - 1}
    snap = {"v": {"response": {"message_id": response_id, "fragments": frags}}}
    return (
        f"event: ready\ndata: {json.dumps(head)}\n\n"
        f"data: {json.dumps(snap, ensure_ascii=False)}\n\n"
        'data: {"p":"response/status","o":"SET","v":"FINISHED"}\n\n'
        "event: close\ndata: {}\n\n"
    ).encode()


class FakeDeepSeek:
    """Подставной сайт: помнит чаты, запросы и отдаёт «ответ N»."""

    def __init__(self) -> None:
        self.chats: list[str] = []
        self.completions: list[dict] = []
        self.deleted: list[str] = []
        self.gone: set[str] = set()
        self.fail_with: httpx.Response | None = None
        self.on_completion = None
        self.answers = 0

    def __call__(self, request: httpx.Request) -> httpx.Response:
        path = request.url.path
        if path == "/api/v0/chat/create_pow_challenge":
            return httpx.Response(200, json=envelope({"challenge": CHALLENGE}))
        if path == "/api/v0/chat_session/create":
            chat_id = f"chat-{len(self.chats) + 1}"
            self.chats.append(chat_id)
            return httpx.Response(200, json=envelope({"chat_session": {"id": chat_id}}))
        if path == "/api/v0/chat_session/delete":
            self.deleted.append(json.loads(request.content)["chat_session_id"])
            return httpx.Response(200, json=envelope({}))
        assert path == "/api/v0/chat/completion", path
        body = json.loads(request.content)
        self.completions.append(body)
        if self.on_completion:
            self.on_completion(body)
        if self.fail_with is not None:
            return self.fail_with
        if body["chat_session_id"] in self.gone:
            return httpx.Response(
                200,
                json=envelope(None, 1, "invalid chat session id"),
            )
        self.answers += 1
        return httpx.Response(
            200,
            content=sse(f"ответ {self.answers}", self.answers * 2, "думаю"),
            headers={"content-type": "text/event-stream"},
        )


class WebCase(unittest.TestCase):
    max_concurrent = 1
    settings: dict = {}

    def setUp(self):
        self.dir = Path(tempfile.mkdtemp())
        self.addCleanup(shutil.rmtree, self.dir, ignore_errors=True)
        account = self.dir / "a.json"
        account.write_text(json.dumps({"token": "T", "wasmUrl": "u"}))
        self.site = FakeDeepSeek()
        self.store = ConversationStore()
        self.context = TransportContext(
            self.store,
            http_factory=lambda proxy: httpx.Client(
                transport=httpx.MockTransport(self.site)
            ),
            pow_solver=lambda challenge, url, http: 1,
        )
        provider = Provider(
            name="dsw",
            api_style="deepseek_web",
            base_url="https://chat.test",
            models_url="",
            auth="web_session",
            web=WebSettings(models=(("m", WebModel()),), **self.settings),
        )
        registry = Registry(
            providers={"dsw": provider},
            proxies={"direct": Proxy(name="direct", address="")},
            keys={
                "DS_A": Key(
                    "DS_A",
                    "dsw",
                    ("direct",),
                    key_file=str(account),
                    max_concurrent=self.max_concurrent,
                )
            },
            models={"m": Model(name="m", providers=("dsw",))},
            env_keys={},
        )
        self.runner = ModelRunner(registry, context=self.context)
        self.addCleanup(self.runner.close)

    def ask(self, messages, namespace="default", session="", **params):
        return self.runner.complete(
            "m",
            messages,
            conversation=ConversationHint(namespace, session),
            **params,
        )

    def states(self) -> dict[str, str]:
        out = {}
        for state in (ACTIVE, BUSY, ABANDONED):
            for record in self.store.by_state(state):
                out[record.chat_id] = state
        return out


SYSTEM = {"role": "system", "content": "Ты помощник."}


def user(text):
    return {"role": "user", "content": text}


def assistant(text):
    return {"role": "assistant", "content": text}


class ConversationTests(WebCase):
    def test_new_conversation_after_another_gets_a_new_chat(self):
        """R2: `[system, user]` без ассистента — всегда новый разговор."""
        first = self.ask([SYSTEM, user("проект A")])
        second = self.ask([SYSTEM, user("совсем другой проект B")])
        self.assertEqual((first.text, second.text), ("ответ 1", "ответ 2"))
        self.assertEqual(self.site.chats, ["chat-1", "chat-2"])
        last = self.site.completions[-1]
        self.assertEqual(last["chat_session_id"], "chat-2")
        self.assertIsNone(last["parent_message_id"])
        self.assertNotIn("проект A", last["prompt"])
        self.assertIn("совсем другой проект B", last["prompt"])
        self.assertIn("Ты помощник.", last["prompt"])

    def test_identical_first_message_is_still_a_new_conversation(self):
        self.ask([SYSTEM, user("привет")])
        self.ask([SYSTEM, user("привет")])
        self.assertEqual(len(self.site.chats), 2)
        self.assertIsNone(self.site.completions[-1]["parent_message_id"])

    def test_continuation_goes_as_delta_to_the_same_chat(self):
        first = self.ask([SYSTEM, user("вопрос 1")])
        second = self.ask(
            [SYSTEM, user("вопрос 1"), assistant(first.text), user("вопрос 2")]
        )
        self.assertEqual(second.text, "ответ 2")
        self.assertEqual(self.site.chats, ["chat-1"])
        sent = self.site.completions[-1]
        self.assertEqual(sent["chat_session_id"], "chat-1")
        self.assertEqual(sent["parent_message_id"], 2)  # response_message_id хода 1
        self.assertEqual(sent["prompt"], "User: вопрос 2")
        self.assertEqual(self.states(), {"chat-1": ACTIVE})

    def test_third_turn_keeps_chaining(self):
        a = self.ask([SYSTEM, user("1")])
        b = self.ask([SYSTEM, user("1"), assistant(a.text), user("2")])
        self.ask(
            [
                SYSTEM,
                user("1"),
                assistant(a.text),
                user("2"),
                assistant(b.text),
                user("3"),
            ]
        )
        self.assertEqual(self.site.chats, ["chat-1"])
        self.assertEqual(self.site.completions[-1]["parent_message_id"], 4)
        self.assertEqual(self.site.completions[-1]["prompt"], "User: 3")

    def test_echo_with_stripped_whitespace_still_continues(self):
        a = self.ask([user("1")])
        self.ask([user("1"), assistant(f"  {a.text}\n"), user("2")])
        self.assertEqual(self.site.chats, ["chat-1"])

    def test_changed_system_starts_new_chat_and_abandons_old(self):
        a = self.ask([SYSTEM, user("1")])
        self.ask(
            [
                {"role": "system", "content": "Другая роль."},
                user("1"),
                assistant(a.text),
                user("2"),
            ]
        )
        self.assertEqual(self.site.chats, ["chat-1", "chat-2"])
        self.assertIn("Другая роль.", self.site.completions[-1]["prompt"])
        self.assertEqual(self.states(), {"chat-1": ABANDONED, "chat-2": ACTIVE})

    def test_rewritten_echo_is_not_a_proven_continuation(self):
        self.ask([user("1")])
        self.ask([user("1"), assistant("совсем другой текст"), user("2")])
        self.assertEqual(len(self.site.chats), 2)
        self.assertIn("совсем другой текст", self.site.completions[-1]["prompt"])

    def test_namespaces_never_share_a_chat(self):
        a = self.ask([user("1")], namespace="hermes")
        self.ask([user("1"), assistant(a.text), user("2")], namespace="other")
        self.assertEqual(len(self.site.chats), 2)
        self.assertIsNone(self.site.completions[-1]["parent_message_id"])

    def test_sessions_never_share_a_chat(self):
        a = self.ask([user("1")], session="s1")
        self.ask([user("1"), assistant(a.text), user("2")], session="s2")
        self.assertEqual(len(self.site.chats), 2)

    def test_only_echo_in_tail_is_not_a_continuation(self):
        a = self.ask([user("1")])
        self.ask([user("1"), assistant(a.text)])
        self.assertEqual(len(self.site.chats), 2)

    def test_default_namespace_without_hint(self):
        a = self.runner.complete("m", [user("1")])
        self.runner.complete("m", [user("1"), assistant(a.text), user("2")])
        self.assertEqual(self.site.chats, ["chat-1"])

    def test_chat_gone_is_transparent(self):
        a = self.ask([SYSTEM, user("1")])
        self.site.gone.add("chat-1")
        reply = self.ask([SYSTEM, user("1"), assistant(a.text), user("2")])
        self.assertTrue(reply.text)
        self.assertEqual(self.site.chats, ["chat-1", "chat-2"])
        retry = self.site.completions[-1]
        self.assertEqual(retry["chat_session_id"], "chat-2")
        self.assertIsNone(retry["parent_message_id"])
        self.assertIn("User: 1", retry["prompt"])  # полный контекст
        self.assertIn("User: 2", retry["prompt"])
        self.assertEqual(self.states(), {"chat-1": ABANDONED, "chat-2": ACTIVE})


class ParallelTests(WebCase):
    max_concurrent = 2

    def test_parallel_request_of_same_conversation_uses_ephemeral_chat(self):
        a = self.ask([SYSTEM, user("1")])
        history = [SYSTEM, user("1"), assistant(a.text), user("2")]
        nested: list = []

        def during(body):
            if not nested and body["chat_session_id"] == "chat-1":
                nested.append(self.ask(history))

        self.site.on_completion = during
        outer = self.ask(history)
        self.assertEqual(len(nested), 1)
        self.assertEqual(self.site.chats, ["chat-1", "chat-2"])
        # Параллельный ход получил полный контекст в новом чате...
        call = next(
            c for c in self.site.completions if c["chat_session_id"] == "chat-2"
        )
        self.assertIsNone(call["parent_message_id"])
        self.assertIn("User: 1", call["prompt"])
        self.assertIn("User: 2", call["prompt"])
        # ...и после ответа ушёл в брошенные, а основной чат остался живым.
        self.assertEqual(self.states(), {"chat-1": ACTIVE, "chat-2": ABANDONED})
        self.assertTrue(outer.text and nested[0].text)


class LimitTests(WebCase):
    settings = {"chat_max_turns": 1}

    def test_turn_limit_rotates_the_chat(self):
        a = self.ask([user("1")])
        self.ask([user("1"), assistant(a.text), user("2")])
        self.assertEqual(len(self.site.chats), 2)
        self.assertEqual(self.states(), {"chat-1": ABANDONED, "chat-2": ACTIVE})


class CharLimitTests(WebCase):
    settings = {"chat_max_chars": 10}

    def test_char_limit_rotates_the_chat(self):
        a = self.ask([user("длинный вопрос номер один")])
        self.ask([user("длинный вопрос номер один"), assistant(a.text), user("2")])
        self.assertEqual(len(self.site.chats), 2)


class SizeTests(WebCase):
    settings = {"inline_max_chars": 50}

    def test_oversized_prompt_is_too_large(self):
        with self.assertRaises(GatewayError) as ctx:
            self.ask([user("x" * 200)])
        self.assertEqual(ctx.exception.attempts[-1].failure, Failure.TOO_LARGE)
        self.assertEqual(self.site.chats, [])  # чат даже не заводили


class FailureTests(WebCase):
    def failing(self, response: httpx.Response):
        self.site.fail_with = response
        with self.assertRaises(GatewayError) as ctx:
            self.ask([user("1")])
        return ctx.exception.attempts[-1]

    def test_backend_busy_pauses_the_binding_and_abandons_the_chat(self):
        body = (
            'event: hint\ndata: {"type":"error","finish_reason":"backend_busy",'
            '"content":"Сервер занят"}\n\n'
        ).encode()
        attempt = self.failing(
            httpx.Response(
                200, content=body, headers={"content-type": "text/event-stream"}
            )
        )
        self.assertEqual(attempt.failure, Failure.QUOTA)
        self.assertEqual(attempt.retry_after, 30.0)
        self.assertEqual(self.states(), {"chat-1": ABANDONED})

    def test_dead_token_marks_account_and_fails_key(self):
        attempt = self.failing(
            httpx.Response(200, json={"code": 40003, "msg": "INVALID_TOKEN"})
        )
        self.assertEqual(attempt.failure, Failure.KEY)
        self.assertEqual(self.store.account_health("DS_A"), "needs_auth")

    def test_success_marks_account_ok(self):
        self.store.mark_account("DS_A", "needs_auth")
        self.ask([user("1")])
        self.assertEqual(self.store.account_health("DS_A"), "ok")

    def test_busy_slot_is_not_a_failure_of_the_account(self):
        transport = self.runner.transport_for(self.runner.registry.provider("dsw"))
        route = self.runner.registry.route("m", "dsw")
        binding = route.bindings[0]
        self.assertTrue(self.context.busy.acquire("DS_A", 1))
        attempt = transport.complete(
            route, binding, ChatRequest("m", [user("1")], conversation=None)
        )
        self.assertEqual(attempt.failure, Failure.QUOTA)
        self.assertEqual(attempt.retry_after, 0.0)
        self.assertEqual(self.site.completions, [])
        # И ключ после этого не ушёл на паузу.
        self.context.busy.release("DS_A")
        self.assertEqual(self.ask([user("2")]).text, "ответ 1")

    def test_slot_is_released_after_failure(self):
        self.failing(httpx.Response(200, json={"code": 40003, "msg": "x"}))
        self.assertEqual(self.context.busy.count("DS_A"), 0)

    def test_tools_are_rejected_until_stage_5(self):
        with self.assertRaises(GatewayError) as ctx:
            self.ask(
                [user("1")], tools=[{"type": "function", "function": {"name": "f"}}]
            )
        self.assertEqual(ctx.exception.attempts[-1].failure, Failure.FATAL)


class StreamTests(WebCase):
    def test_stream_yields_text_and_commits_chat(self):
        with self.runner.stream(
            "m", [user("1")], conversation=ConversationHint("default", "")
        ) as stream:
            chunks = list(stream)
            response = stream.collect()
        self.assertEqual("".join(c.delta for c in chunks), "ответ 1")
        self.assertEqual(chunks[-1].finish_reason, "stop")
        self.assertEqual(response.text, "ответ 1")
        self.assertEqual(response.reasoning, "думаю")
        self.assertEqual(self.states(), {"chat-1": ACTIVE})
        # Продолжение после стрима идёт дельтой.
        self.ask([user("1"), assistant("ответ 1"), user("2")])
        self.assertEqual(self.site.chats, ["chat-1"])

    def test_closing_stream_early_abandons_the_chat(self):
        stream = self.runner.stream("m", [user("1")])
        stream.close()
        self.assertEqual(self.states(), {"chat-1": ABANDONED})
        self.assertEqual(self.context.busy.count("DS_A"), 0)

    def test_failure_before_first_chunk_lets_rotation_fail_over(self):
        self.site.fail_with = httpx.Response(200, json={"code": 40003, "msg": "x"})
        with self.assertRaises(GatewayError):
            self.runner.stream("m", [user("1")])


class FingerprintTests(unittest.TestCase):
    def calls(self, call_id):
        return {
            "role": "assistant",
            "content": None,
            "tool_calls": [
                {
                    "id": call_id,
                    "type": "function",
                    "function": {"name": "ls", "arguments": '{"b": 1, "a": 2}'},
                }
            ],
        }

    def test_rewritten_tool_call_ids_do_not_change_keys(self):
        one = [
            user("x"),
            self.calls("call_1"),
            {"role": "tool", "tool_call_id": "call_1", "content": "ok"},
        ]
        two = [
            user("x"),
            self.calls("toolu_ZZ"),
            {"role": "tool", "tool_call_id": "toolu_ZZ", "content": "ok"},
        ]
        self.assertEqual(
            fingerprint.fingerprint(one).request_keys,
            fingerprint.fingerprint(two).request_keys,
        )

    def test_argument_order_does_not_matter(self):
        a = self.calls("1")
        b = self.calls("1")
        b["tool_calls"][0]["function"]["arguments"] = '{"a": 2, "b": 1}'
        self.assertEqual(fingerprint.message_key(a), fingerprint.message_key(b))

    def test_different_text_different_key(self):
        self.assertNotEqual(
            fingerprint.message_key(user("a")), fingerprint.message_key(user("b"))
        )

    def test_reasoning_is_ignored(self):
        plain = assistant("hi")
        with_reasoning = {**plain, "reasoning_content": "думал"}
        self.assertEqual(
            fingerprint.message_key(plain), fingerprint.message_key(with_reasoning)
        )

    def test_context_key_depends_on_system_and_tools(self):
        base = fingerprint.fingerprint([SYSTEM, user("x")]).context_key
        self.assertEqual(base, fingerprint.fingerprint([SYSTEM, user("y")]).context_key)
        self.assertNotEqual(
            base,
            fingerprint.fingerprint(
                [{"role": "system", "content": "иное"}, user("x")]
            ).context_key,
        )
        tools = [{"type": "function", "function": {"name": "f", "parameters": {}}}]
        self.assertNotEqual(
            base, fingerprint.fingerprint([SYSTEM, user("x")], tools).context_key
        )

    def test_reply_key_matches_client_echo(self):
        self.assertEqual(
            fingerprint.reply_key("текст"), fingerprint.message_key(assistant("текст"))
        )


class PlannerTests(unittest.TestCase):
    """Планировщик на подготовленном хранилище, без сети."""

    def setUp(self):
        self.store = ConversationStore()
        self.settings = WebSettings()
        self.messages = [SYSTEM, user("1"), assistant("a1"), user("2")]
        fp = fingerprint.fingerprint(self.messages[:2])
        self.sent = (*fp.request_keys, fingerprint.reply_key("a1"))
        self.context_key = fp.context_key

    def seed(self, state=ACTIVE, **overrides):
        self.store.create_chat(
            "dsw", "c1", "K", "direct", "default", "", self.context_key, state
        )
        self.store.finish_turn("dsw", "c1", self.sent, 7, 10, state)

    def plan(self, messages=None, **kw):
        return plan_turn(
            self.store,
            self.settings,
            provider="dsw",
            key_name="K",
            namespace="default",
            session="",
            messages=messages or self.messages,
            **kw,
        )

    def test_delta_claims_the_chat(self):
        self.seed()
        plan = self.plan()
        self.assertEqual(plan.kind, "delta")
        self.assertEqual(plan.chat.chat_id, "c1")
        self.assertEqual(self.store.get("dsw", "c1").state, BUSY)

    def test_busy_chat_gives_ephemeral_full(self):
        self.seed(BUSY)
        plan = self.plan()
        self.assertEqual((plan.kind, plan.ephemeral), ("full", True))

    def test_other_key_never_reuses_chat(self):
        self.seed()
        plan = plan_turn(
            self.store,
            self.settings,
            provider="dsw",
            key_name="OTHER",
            namespace="default",
            session="",
            messages=self.messages,
        )
        self.assertEqual(plan.kind, "full")

    def test_no_chat_means_full_prompt_with_system(self):
        plan = self.plan()
        self.assertEqual(plan.kind, "full")
        self.assertTrue(plan.prompt.startswith("Ты помощник."))
        self.assertIn("Assistant: a1", plan.prompt)


class BusyTrackerTests(unittest.TestCase):
    def test_limit(self):
        busy = BusyTracker()
        self.assertTrue(busy.acquire("k", 1))
        self.assertFalse(busy.acquire("k", 1))
        busy.release("k")
        self.assertTrue(busy.acquire("k", 1))

    def test_unlimited(self):
        busy = BusyTracker()
        self.assertTrue(all(busy.acquire("k", None) for _ in range(10)))


class StoreTests(unittest.TestCase):
    def test_claim_is_exclusive(self):
        store = ConversationStore()
        store.create_chat("p", "c", "K", "direct", "ns", "", "ctx", ACTIVE)
        self.assertTrue(store.claim("p", "c"))
        self.assertFalse(store.claim("p", "c"))

    def test_persists_on_disk(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "state" / "rotator.db"
            store = ConversationStore(path)
            store.create_chat("p", "c", "K", "direct", "ns", "", "ctx", ACTIVE)
            store.finish_turn("p", "c", ["a", "b"], 5, 3)
            store.close()
            record = ConversationStore(path).get("p", "c")
            self.assertEqual(record.sent_keys, ("a", "b"))
            self.assertEqual(record.parent_message_id, 5)
            self.assertEqual(record.turns, 1)


if __name__ == "__main__":
    unittest.main()
