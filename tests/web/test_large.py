"""R4: большой контекст файлом и картинки."""

from __future__ import annotations

import base64
import unittest

import httpx

from llm_rotator import Failure, GatewayError
from llm_rotator.web.planner import CONTEXT_NOTE, TRUNCATION_MARKER

from .test_transport import SYSTEM, WebCase, assistant, user


def png(seed: int = 0) -> dict:
    data = b"\x89PNG\r\n\x1a\n" + bytes([seed]) * 32
    url = "data:image/png;base64," + base64.b64encode(data).decode()
    return {"type": "image_url", "image_url": {"url": url}}


def last_completion(case: WebCase) -> dict:
    return case.site.completions[-1]


class LargeContextTests(WebCase):
    settings = {"inline_max_chars": 50_000, "file_max_chars": 400_000}

    def test_big_history_goes_to_context_file(self):
        history = [
            SYSTEM,
            user("НАЧАЛО " + "а" * 30_000),
            assistant("б" * 30_000 + " КОНЕЦ"),
            user("итоговый вопрос"),
        ]
        self.ask(history)
        sent = last_completion(self)
        self.assertLess(len(sent["prompt"]), 2_000)
        self.assertIn(CONTEXT_NOTE, sent["prompt"])
        self.assertIn("User: итоговый вопрос", sent["prompt"])
        self.assertIn("Ты помощник.", sent["prompt"])
        self.assertEqual(sent["ref_file_ids"], ["file-1"])
        (upload,) = self.site.uploads
        self.assertEqual(upload["name"], "context.md")
        self.assertIn("НАЧАЛО".encode(), upload["body"])
        self.assertIn("КОНЕЦ".encode(), upload["body"])
        self.assertNotIn("итоговый вопрос".encode(), upload["body"])

    def test_300k_single_message_is_attached_as_file(self):
        text = "ПЕРВАЯ " + "в" * 300_000 + " ПОСЛЕДНЯЯ"
        self.ask([SYSTEM, user(text)])
        sent = last_completion(self)
        self.assertLess(len(sent["prompt"]), 2_000)
        self.assertRegex(sent["prompt"], r"attachment_[0-9a-f]{8}\.txt")
        self.assertEqual(len(sent["ref_file_ids"]), 1)
        body = self.site.uploads[0]["body"]
        self.assertIn("ПЕРВАЯ".encode(), body)
        self.assertIn("ПОСЛЕДНЯЯ".encode(), body)

    def test_small_context_stays_inline(self):
        self.ask([SYSTEM, user("коротко")])
        self.assertEqual(self.site.uploads, [])
        self.assertEqual(last_completion(self)["ref_file_ids"], [])

    def test_identical_file_is_uploaded_once_per_account(self):
        big = [user("Z" * 60_000), assistant("ок"), user("вопрос")]
        self.ask(big)
        self.ask(big, namespace="other")
        self.assertEqual(len(self.site.uploads), 1)
        self.assertEqual(last_completion(self)["ref_file_ids"], ["file-1"])

    def test_big_tool_result_in_a_delta_turn_is_attached_without_history_file(self):
        first = self.ask([user("прочти")])
        history = [
            user("прочти"),
            assistant(first.text),
            {"role": "user", "content": "вот файл:\n" + "д" * 80_000},
        ]
        self.ask(history)
        self.assertEqual(self.site.chats, ["chat-1"])  # тот же чат
        sent = last_completion(self)
        self.assertEqual(sent["parent_message_id"], 2)
        (upload,) = self.site.uploads
        self.assertRegex(upload["name"], r"attachment_[0-9a-f]{8}\.txt")
        self.assertLess(len(sent["prompt"]), 2_000)

    def test_chat_gone_resends_full_context_with_files(self):
        big = [user("Y" * 60_000)]
        first = self.ask(big)
        self.site.gone.add("chat-1")
        self.ask([*big, assistant(first.text), user("дальше")])
        retry = last_completion(self)
        self.assertEqual(retry["chat_session_id"], "chat-2")
        self.assertEqual(len(retry["ref_file_ids"]), 1)
        self.assertIn("User: дальше", retry["prompt"])

    def test_oversized_system_is_too_large_before_any_chat(self):
        with self.assertRaises(GatewayError) as ctx:
            self.ask([{"role": "system", "content": "s" * 60_000}, user("x")])
        self.assertEqual(ctx.exception.attempts[-1].failure, Failure.TOO_LARGE)
        self.assertEqual((self.site.chats, self.site.uploads), ([], []))

    def test_upload_failure_is_a_pause_not_a_fatal_error(self):
        self.site.upload_error = httpx.Response(429, json={})
        with self.assertRaises(GatewayError) as ctx:
            self.ask([user("Q" * 60_000)])
        self.assertEqual(ctx.exception.attempts[-1].failure, Failure.QUOTA)


class TruncationTests(WebCase):
    settings = {"inline_max_chars": 50_000, "file_max_chars": 100_000}

    def test_file_over_limit_keeps_head_and_tail(self):
        text = "ГОЛОВА " + "м" * 300_000 + " ХВОСТ"
        self.ask([user(text)])
        body = self.site.uploads[0]["body"]
        self.assertLess(len(body), 100_000 * 2 + 1_000)  # байты: кириллица — 2 байта
        self.assertIn(TRUNCATION_MARKER.strip().encode(), body)
        self.assertIn("ГОЛОВА".encode(), body)
        self.assertIn("ХВОСТ".encode(), body)


class ImageTests(WebCase):
    settings = {"images": True}

    def message(self, *images, text="что на картинке?"):
        return {"role": "user", "content": [{"type": "text", "text": text}, *images]}

    def test_images_are_uploaded_and_referenced(self):
        self.ask([self.message(png(1))])
        sent = last_completion(self)
        self.assertEqual(sent["ref_file_ids"], ["file-1"])
        self.assertTrue(self.site.uploads[0]["name"].startswith("image-"))
        self.assertTrue(self.site.uploads[0]["name"].endswith(".png"))
        self.assertIn("что на картинке?", sent["prompt"])

    def test_only_the_newest_four_images_are_sent(self):
        self.ask([self.message(*[png(i) for i in range(6)])])
        self.assertEqual(len(self.site.uploads), 4)

    def test_repeated_image_is_uploaded_once(self):
        self.ask([self.message(png(1), png(1))])
        self.assertEqual(len(self.site.uploads), 1)

    def test_remote_urls_are_not_fetched(self):
        remote = {
            "type": "image_url",
            "image_url": {"url": "https://example.com/a.png"},
        }
        self.ask([self.message(remote)])
        self.assertEqual(self.site.uploads, [])

    def test_unsupported_mime_is_not_uploaded(self):
        svg = {
            "type": "image_url",
            "image_url": {"url": "data:image/svg+xml;base64,PHN2Zy8+"},
        }
        self.ask([self.message(svg)])
        self.assertEqual(self.site.uploads, [])

    def test_continuation_with_image_in_history_is_still_a_delta(self):
        first = self.ask([self.message(png(1))])
        self.ask([self.message(png(1)), assistant(first.text), user("а теперь?")])
        self.assertEqual(self.site.chats, ["chat-1"])
        self.assertEqual(
            len(self.site.uploads), 1
        )  # старая картинка не грузится заново


class ImagesOffTests(WebCase):
    def test_disabled_images_become_an_honest_marker(self):
        content = [{"type": "text", "text": "смотри"}, png(1)]
        self.ask([{"role": "user", "content": content}])
        self.assertEqual(self.site.uploads, [])
        self.assertIn("not visible through this proxy", last_completion(self)["prompt"])


if __name__ == "__main__":
    unittest.main()
