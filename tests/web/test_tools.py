"""Эмуляция tool calling: разбор ответа (сверка с JS-эталоном) и сквозные ходы."""

from __future__ import annotations

import json
import unittest
from pathlib import Path

from llm_rotator import ConversationHint
from llm_rotator.web import tools as T

from .test_transport import WebCase, assistant, user

FIXTURE = json.loads(
    (
        Path(__file__).resolve().parent.parent / "fixtures/deepseek/tools_parse.json"
    ).read_text()
)

TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "read_file",
            "description": "Читает файл",
            "parameters": {
                "type": "object",
                "properties": {"path": {"type": "string", "description": "путь"}},
                "required": ["path"],
            },
        },
    }
]
CALL = '{"tool_call":{"name":"read_file","arguments":{"path":"/tmp/a"}}}'


class ReferenceParserTests(unittest.TestCase):
    """Те же входы, что и у parseToolCall из FreeDeepseekAPI: те же ответы."""

    def test_parse_matches_reference(self):
        for name, case in FIXTURE["parse"].items():
            with self.subTest(name):
                call = T.parse_tool_call(case["input"])
                got = {"name": call.name, "arguments": call.arguments} if call else None
                self.assertEqual(got, case["call"])
                self.assertEqual(
                    T.looks_like_tool_call_markup(case["input"]), case["markup"]
                )

    def test_fixture_is_not_trivial(self):
        found = [c for c in FIXTURE["parse"].values() if c["call"]]
        self.assertGreater(len(found), 15)
        self.assertGreater(len(FIXTURE["parse"]) - len(found), 15)

    def test_lenient_json_matches_reference(self):
        for case in FIXTURE["lenient"]:
            with self.subTest(case["input"]):
                try:
                    got = {"ok": T.parse_json_lenient(case["input"])}
                except ValueError:
                    got = {"error": True}
                self.assertEqual(got, {k: v for k, v in case.items() if k != "input"})

    def test_schema_compaction_matches_reference(self):
        for case in FIXTURE["schemas"]:
            self.assertEqual(T.compact_tool_schema(case["input"]), case["expected"])

    def test_unknown_tool_name_is_not_a_call(self):
        call = T.parse_tool_call('{"tool_call":{"name":"rm_rf","arguments":{}}}')
        self.assertIsNone(T.known_call(call, TOOLS))
        self.assertIsNotNone(T.known_call(T.parse_tool_call(CALL), TOOLS))


class PromptTests(unittest.TestCase):
    def test_definitions_list_functions_and_format(self):
        text = T.format_tool_definitions(TOOLS, 200_000)
        self.assertIn("## read_file", text)
        self.assertIn(T.TOOL_CALL_FORMAT, text)
        self.assertIn("Читает файл", text)
        self.assertIn('"path"', text)

    def test_big_schemas_are_compacted(self):
        text = T.format_tool_definitions(TOOLS, 50)
        self.assertNotIn("путь", text)
        self.assertIn('"path"', text)

    def test_no_tools_no_text(self):
        self.assertEqual(T.format_tool_definitions(None, 1000), "")
        self.assertEqual(T.format_tool_reminder([]), "")

    def test_reminder_names_tools(self):
        self.assertIn("read_file", T.format_tool_reminder(TOOLS))


def tool_call_message(call_id: str, args: dict) -> dict:
    return {
        "role": "assistant",
        "content": None,
        "tool_calls": [
            {
                "id": call_id,
                "type": "function",
                "function": {"name": "read_file", "arguments": json.dumps(args)},
            }
        ],
    }


class ToolFlowTests(WebCase):
    def test_tool_call_comes_back_as_toolcall(self):
        self.site.replies = ["Сначала прочитаю.\n" + CALL]
        response = self.ask([user("прочти /tmp/a")], tools=TOOLS)
        self.assertEqual(response.finish_reason, "tool_calls")
        self.assertEqual(response.text, "")
        (call,) = response.tool_calls
        self.assertEqual(
            (call.name, dict(call.arguments)), ("read_file", {"path": "/tmp/a"})
        )
        self.assertTrue(call.id.startswith("call_"))
        prompt = self.site.completions[-1]["prompt"]
        self.assertIn("TOOL REQUEST SYSTEM", prompt)
        self.assertIn("## read_file", prompt)

    def test_plain_answer_with_tools_offered_is_text(self):
        self.site.replies = ["Всё готово."]
        response = self.ask([user("привет")], tools=TOOLS)
        self.assertEqual((response.text, response.tool_calls), ("Всё готово.", ()))
        self.assertEqual(response.finish_reason, "stop")

    def test_tool_result_continues_the_same_chat_with_reminder(self):
        self.site.replies = [CALL, "В файле: привет"]
        first = self.ask([user("прочти /tmp/a")], tools=TOOLS)
        # Клиент переписал id вызова — продолжение всё равно доказано.
        history = [
            user("прочти /tmp/a"),
            tool_call_message("toolu_REWRITTEN", dict(first.tool_calls[0].arguments)),
            {"role": "tool", "tool_call_id": "toolu_REWRITTEN", "content": "привет"},
        ]
        second = self.ask(history, tools=TOOLS)
        self.assertEqual(second.text, "В файле: привет")
        self.assertEqual(self.site.chats, ["chat-1"])
        sent = self.site.completions[-1]
        self.assertEqual(sent["parent_message_id"], 2)
        self.assertIn("[Tool Result: read_file]\nпривет", sent["prompt"])
        self.assertIn("[Tool reminder]", sent["prompt"])
        self.assertNotIn("TOOL REQUEST SYSTEM", sent["prompt"])

    def test_malformed_markup_is_retried_in_the_same_chat(self):
        broken = '{"tool_call":{"name":"read_file","arguments":{"path":"/tmp/a"'
        self.site.replies = [broken, CALL]
        response = self.ask([user("прочти")], tools=TOOLS)
        self.assertEqual(response.tool_calls[0].name, "read_file")
        self.assertEqual(self.site.chats, ["chat-1"])  # нового чата не заводили
        retry = self.site.completions[-1]
        self.assertEqual(retry["chat_session_id"], "chat-1")
        self.assertEqual(retry["parent_message_id"], 2)
        self.assertIn("malformed tool call", retry["prompt"])

    def test_hallucinated_tool_name_stays_text(self):
        bad = '{"tool_call":{"name":"format_disk","arguments":{}}}'
        self.site.replies = [bad, bad]
        response = self.ask([user("x")], tools=TOOLS)
        self.assertEqual(response.tool_calls, ())
        self.assertIn("format_disk", response.text)
        self.assertEqual(len(self.site.completions), 2)  # одно уточнение, не больше
        self.assertEqual(self.site.chats, ["chat-1"])

    def test_tool_choice_none_hides_tools(self):
        self.ask([user("x")], tools=TOOLS, tool_choice="none")
        self.assertNotIn("TOOL REQUEST SYSTEM", self.site.completions[-1]["prompt"])

    def test_changed_tools_start_a_new_chat(self):
        first = self.ask([user("x")], tools=TOOLS)
        self.ask([user("x"), assistant(first.text), user("y")], tools=[])
        self.assertEqual(len(self.site.chats), 2)

    def test_streaming_with_tools_is_buffered(self):
        self.site.replies = [CALL]
        with self.runner.stream(
            "m", [user("прочти")], tools=TOOLS, conversation=ConversationHint()
        ) as stream:
            chunks = list(stream)
            response = stream.collect()
        self.assertEqual("".join(c.delta for c in chunks), "")
        self.assertEqual(chunks[-1].finish_reason, "tool_calls")
        self.assertEqual(response.tool_calls[0].name, "read_file")


if __name__ == "__main__":
    unittest.main()
