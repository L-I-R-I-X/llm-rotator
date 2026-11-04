"""Аккаунты без ручной правки keys.toml: sync, auth с именем и цикл входов."""

from __future__ import annotations

import argparse
import contextlib
import io
import json
import os
import stat
import unittest
from unittest import mock

from llm_rotator import accounts, cli, load_registry
from llm_rotator.web.deepseek import auth as login

from .test_cli import Base

GOOD = login.SessionData(token="TOK", cookie="c=1", cookies_count=1, href="https://x")
EMPTY = login.SessionData(token="", cookie="")


class NameTests(unittest.TestCase):
    def test_names(self):
        self.assertEqual(accounts.normalize_name(" me@gmail.com "), "me@gmail.com")
        self.assertEqual(accounts.normalize_name("work.json"), "work")
        for bad in ("", "a/b", "../x", "-x", "a b", "a..b", "x" * 65, "ключ"):
            with self.assertRaises(ValueError, msg=bad):
                accounts.normalize_name(bad)


class SyncTests(Base):
    def registry(self):
        return load_registry(self.dir, check_key_files=False)

    def run_sync(self, **extra):
        args = argparse.Namespace(
            base_dir=self.dir, provider=None, dry_run=False, **extra
        )
        out, err = io.StringIO(), io.StringIO()
        with contextlib.redirect_stdout(out), contextlib.redirect_stderr(err):
            code = cli._accounts_sync(args)
        return code, out.getvalue(), err.getvalue()

    def test_sync_adds_only_unknown_files_and_is_idempotent(self):
        (self.dir / "accounts" / "me@gmail.com.json").write_text("{}")
        (self.dir / "accounts" / "bad name.json").write_text("{}")
        code, out, err = self.run_sync()
        self.assertEqual(code, 2)  # один файл пропущен из-за имени
        self.assertIn("me@gmail.com", out)
        self.assertIn("bad name.json", err)
        keys = self.registry().keys
        self.assertEqual(
            keys["me@gmail.com"].key_file,
            str(self.dir / "accounts" / "me@gmail.com.json"),
        )
        self.assertEqual(keys["me@gmail.com"].proxies, ("direct",))
        self.assertEqual(keys["me@gmail.com"].max_concurrent, 1)
        before = (self.dir / "models" / "keys.toml").read_text()
        self.run_sync()
        self.assertEqual((self.dir / "models" / "keys.toml").read_text(), before)

    def test_name_clash_gets_auto_name_and_dry_run_changes_nothing(self):
        (self.dir / "accounts" / "WEB_OK.json").write_text(
            "{}"
        )  # ключ с таким именем есть
        before = (self.dir / "models" / "keys.toml").read_text()
        args = argparse.Namespace(base_dir=self.dir, provider=None, dry_run=True)
        with contextlib.redirect_stdout(io.StringIO()):
            self.assertEqual(cli._accounts_sync(args), 0)
        self.assertEqual((self.dir / "models" / "keys.toml").read_text(), before)
        self.run_sync()
        self.assertIn("DS_1", self.registry().keys)

    def test_nothing_to_do(self):
        self.run_sync()
        _, out, _ = self.run_sync()
        self.assertIn("уже есть", out)


class EnrollTests(Base):
    def args(self, name=None):
        return argparse.Namespace(
            site="deepseek", name=name, base_dir=self.dir, headless=False, provider=None
        )

    def run_auth(self, name, sessions, answers):
        """`answers` — ответы по порядку; KeyboardInterrupt в списке = Ctrl+C."""
        out, err = io.StringIO(), io.StringIO()
        replies = iter(answers)
        prompts: list[str] = []

        def prompt(text):
            prompts.append(text)
            reply = next(replies)
            if reply is KeyboardInterrupt:
                raise KeyboardInterrupt
            return reply

        with (
            contextlib.redirect_stdout(out),
            contextlib.redirect_stderr(err),
            mock.patch.object(login, "capture", side_effect=sessions) as capture,
            mock.patch.object(cli, "_verify", return_value=0),
        ):
            code = cli._auth(self.args(name), prompt=prompt, chrome="/usr/bin/chromium")
        return code, out.getvalue(), err.getvalue(), capture, prompts

    def keys(self):
        return load_registry(self.dir, check_key_files=False).keys

    def test_named_new_account_creates_file_and_key(self):
        code, out, err, capture, _ = self.run_auth("me@gmail.com", [GOOD], [])
        self.assertEqual(code, 0, err)
        path = self.dir / "accounts" / "me@gmail.com.json"
        self.assertEqual(json.loads(path.read_text())["token"], "TOK")
        self.assertEqual(stat.S_IMODE(os.stat(path).st_mode), 0o600)
        self.assertIn("me@gmail.com", self.keys())
        self.assertNotIn("TOK", out + err)
        # браузер: свой профиль, не чужой
        self.assertTrue(
            str(capture.call_args.kwargs["profile"]).endswith("chrome/me@gmail.com")
        )

    def test_named_existing_key_is_relogin_not_a_new_account(self):
        before = set(self.keys())
        code, *_ = self.run_auth("WEB_OK", [GOOD], [])
        self.assertEqual(code, 0)
        self.assertEqual(set(self.keys()), before)
        self.assertEqual(
            json.loads((self.dir / "accounts" / "ok.json").read_text())["token"], "TOK"
        )

    def test_loop_names_default_names_and_ctrl_c(self):
        # 1-й круг: имя «alice»; 2-й: пустое → DS_1; 3-й: Ctrl+C на «ещё один?»
        code, out, err, capture, prompts = self.run_auth(
            None,
            [GOOD, GOOD],
            ["alice", "", "", KeyboardInterrupt],
        )
        self.assertEqual(code, 0, err)
        self.assertEqual(capture.call_count, 2)
        self.assertIn("alice", self.keys())
        self.assertIn("DS_1", self.keys())
        self.assertTrue((self.dir / "accounts" / "DS_1.json").exists())
        self.assertIn("(Enter — DS_1)", "".join(prompts))
        self.assertIn("Добавлено аккаунтов: 2", out)

    def test_loop_profiles_are_fresh_and_removed(self):
        _, _, _, capture, _ = self.run_auth(None, [GOOD], ["", KeyboardInterrupt])
        profile = capture.call_args.kwargs["profile"]
        self.assertIn("_new-", str(profile))
        self.assertFalse(profile.exists())

    def test_name_is_asked_only_after_successful_login(self):
        code, _, err, _, prompts = self.run_auth(None, [EMPTY], [KeyboardInterrupt])
        self.assertEqual(code, 2)
        self.assertNotIn("Имя аккаунта", "".join(prompts))
        self.assertIn("не завершён", err)
        self.assertFalse(list((self.dir / "accounts").glob("DS_*.json")))

    def test_taken_name_is_asked_again(self):
        code, out, _, _, _ = self.run_auth(
            None, [GOOD], ["WEB_OK", "bad/name", "fresh", KeyboardInterrupt]
        )
        self.assertEqual(code, 0)
        self.assertIn("уже есть", out)
        self.assertIn("недопустимое имя", out)
        self.assertIn("fresh", self.keys())

    def test_ctrl_c_during_login_ends_loop_cleanly(self):
        out = io.StringIO()
        with (
            contextlib.redirect_stdout(out),
            mock.patch.object(login, "capture", side_effect=KeyboardInterrupt),
        ):
            code = cli._auth(self.args(), prompt=input, chrome="/usr/bin/chromium")
        self.assertEqual(code, 0)
        self.assertIn("Ни один аккаунт не добавлен", out.getvalue())


if __name__ == "__main__":
    unittest.main()
