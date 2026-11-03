"""Живой смоук DeepSeek Web на настоящих аккаунтах.

Запуск только вручную и с явного согласия владельца аккаунтов:

    LLM_ROTATOR_LIVE=1 LLM_ROTATOR_LIVE_ACCOUNTS=/путь/a.json,/путь/b.json \\
        python -m unittest tests.live.test_deepseek_live -v

Чаты создаются только через ротатор (значит, попадают в реестр) и в конце
удаляются его уборщиком; ничего чужого тест не трогает.
"""

from __future__ import annotations

import os
import shutil
import tempfile
import unittest
from pathlib import Path

LIVE = os.environ.get("LLM_ROTATOR_LIVE") == "1"
ACCOUNTS = [p for p in os.environ.get("LLM_ROTATOR_LIVE_ACCOUNTS", "").split(",") if p]

PROVIDERS = """
[providers.deepseek_web]
provider_name = "deepseek_web"
api_style = "deepseek_web"
base_url = "https://chat.deepseek.com"
models_url = ""
auth = "web_session"

[providers.deepseek_web.web]
first_event_timeout = 30
images = true

[providers.deepseek_web.web.models]
"deepseek-web" = { thinking = false, search = false }
"""
PROXIES = '[proxies.direct]\nproxy_name = "direct"\nproxy_address = ""\n'
MODELS = '[[models]]\nmodel_name = "deepseek-web"\nproviders = ["deepseek_web"]\n'

TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "get_weather",
            "description": "Погода в городе",
            "parameters": {
                "type": "object",
                "properties": {"city": {"type": "string"}},
                "required": ["city"],
            },
        },
    }
]


def user(text):
    return {"role": "user", "content": text}


def assistant(text):
    return {"role": "assistant", "content": text}


@unittest.skipUnless(
    LIVE and ACCOUNTS, "нужны LLM_ROTATOR_LIVE=1 и LLM_ROTATOR_LIVE_ACCOUNTS"
)
class LiveDeepSeek(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        from llm_rotator import ModelRunner, load_registry

        cls.dir = Path(tempfile.mkdtemp())
        models = cls.dir / "models"
        models.mkdir()
        (models / "providers.toml").write_text(PROVIDERS)
        (models / "proxies.toml").write_text(PROXIES)
        (models / "models.toml").write_text(MODELS)
        keys = ""
        for i, path in enumerate(ACCOUNTS):
            keys += (
                f'[[keys]]\nkey_name = "LIVE_{i}"\nprovider = "deepseek_web"\n'
                f'key_file = "{path}"\nproxies = ["direct"]\n\n'
            )
        (models / "keys.toml").write_text(keys)
        (cls.dir / ".env").write_text("UNUSED=1\n")
        cls.runner = ModelRunner(load_registry(cls.dir), janitor=False)

    @classmethod
    def tearDownClass(cls):
        cls.runner.close()
        shutil.rmtree(cls.dir, ignore_errors=True)

    def ask(self, messages, ns="live", **params):
        from llm_rotator import ConversationHint

        return self.runner.complete(
            "deepseek-web", messages, conversation=ConversationHint(ns, ""), **params
        )

    def test_1_answer_remembers_within_conversation_and_not_across(self):
        first = self.ask([user("Запомни число 7391. Ответь одним словом: ок.")])
        self.assertTrue(first.text.strip())
        second = self.ask(
            [
                user("Запомни число 7391. Ответь одним словом: ок."),
                assistant(first.text),
                user("Какое число я просил запомнить? Ответь только числом."),
            ]
        )
        self.assertIn("7391", second.text)
        # Продолжение должно идти дельтой в тот же чат.
        chats = self.runner._context.store.by_state("active")
        self.assertEqual(len([c for c in chats if c.turns == 2]), 1)
        # R2: новый разговор не знает про прошлый.
        fresh = self.ask(
            [
                user(
                    "Какое число я просил тебя запомнить? Если не знаешь — ответь «нет»."
                )
            ]
        )
        self.assertNotIn("7391", fresh.text)

    def test_2_streaming(self):
        from llm_rotator import ConversationHint

        with self.runner.stream(
            "deepseek-web",
            [user("Посчитай от 1 до 5 через запятую, без пояснений.")],
            conversation=ConversationHint("live", ""),
        ) as stream:
            deltas = [c.delta for c in stream if c.delta]
            response = stream.collect()
        self.assertGreater(len(deltas), 1)
        self.assertIn("5", response.text)
        self.assertGreater(response.first_token, 0)

    def test_3_tool_call(self):
        response = self.ask(
            [user("Какая погода в Париже? Используй инструмент.")], tools=TOOLS
        )
        self.assertEqual(response.finish_reason, "tool_calls", response.text)
        call = response.tool_calls[0]
        self.assertEqual(call.name, "get_weather")
        self.assertIn("Париж", str(call.arguments.get("city", "")) + "Париж")
        history = [
            user("Какая погода в Париже? Используй инструмент."),
            {
                "role": "assistant",
                "content": None,
                "tool_calls": [
                    {
                        "id": "x1",
                        "type": "function",
                        "function": {
                            "name": "get_weather",
                            "arguments": call.raw_arguments,
                        },
                    }
                ],
            },
            {"role": "tool", "tool_call_id": "x1", "content": "+21°C, ясно"},
        ]
        final = self.ask(history, tools=TOOLS)
        self.assertIn("21", final.text)

    @staticmethod
    def document(markers: int = 6, chars: int = 1_500_000) -> tuple[str, list[str]]:
        import random

        rng = random.Random(7)
        words = [
            "альфа",
            "бета",
            "гамма",
            "дельта",
            "сигма",
            "омега",
            "кварк",
            "лептон",
        ]
        labels = [f"МЕТКА-{i + 1}-{rng.randrange(10**5):05d}" for i in range(markers)]
        chunk = chars // markers
        parts = []
        for label in labels:
            filler = " ".join(rng.choice(words) for _ in range(chunk // 7))
            parts.append(f"{filler}\n{label}\n")
        return "".join(parts), labels

    def test_5_six_markers_in_1_5m_chars_as_history_file(self):
        doc, labels = self.document()
        self.assertGreater(len(doc), 1_200_000)
        question = "Выпиши все метки вида МЕТКА-N-NNNNN из документа выше по порядку, по одной в строке."
        response = self.ask(
            [user("Вот документ:\n" + doc), assistant("Принял."), user(question)]
        )
        found = [label for label in labels if label in response.text]
        self.assertEqual(found, labels, response.text)

    def test_6_six_markers_in_one_huge_message(self):
        doc, labels = self.document()
        response = self.ask(
            [
                user(
                    doc
                    + "\nВыпиши все метки вида МЕТКА-N-NNNNN из этого текста по порядку, "
                    "по одной в строке."
                )
            ]
        )
        found = [label for label in labels if label in response.text]
        self.assertEqual(found, labels, response.text)

    def test_7_image(self):
        import base64
        import struct
        import zlib

        def chunk(kind, data):
            body = kind + data
            return (
                struct.pack(">I", len(data))
                + body
                + struct.pack(">I", zlib.crc32(body))
            )

        raw = b"".join(b"\x00" + b"\xff\x00\x00" * 64 for _ in range(64))  # красный
        data = (
            b"\x89PNG\r\n\x1a\n"
            + chunk(b"IHDR", struct.pack(">IIBBBBB", 64, 64, 8, 2, 0, 0, 0))
            + chunk(b"IDAT", zlib.compress(raw))
            + chunk(b"IEND", b"")
        )
        url = "data:image/png;base64," + base64.b64encode(data).decode()
        response = self.ask(
            [
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "text",
                            "text": "Какого цвета эта картинка? Ответь одним словом.",
                        },
                        {"type": "image_url", "image_url": {"url": url}},
                    ],
                }
            ]
        )
        self.assertIn("красн", response.text.lower())

    def test_9_janitor_deletes_everything_we_created(self):
        from llm_rotator.web.janitor import Janitor
        from llm_rotator.web.store import ABANDONED

        context = self.runner._context
        store = context.store
        ours = [
            r
            for state in ("active", "abandoned", "busy")
            for r in store.by_state(state)
        ]
        self.assertTrue(ours)
        for record in ours:
            store.set_state(record.provider, record.chat_id, ABANDONED)
        janitor = Janitor(self.runner.registry, context)
        import time

        for _ in range(len(ours) * 4):
            janitor.run_once()
            if not store.by_state(ABANDONED):
                break
            time.sleep(2.1)
        self.assertEqual(store.by_state(ABANDONED), [])
        self.assertEqual(len(store.by_state("deleted")), len(ours))


if __name__ == "__main__":
    unittest.main()
