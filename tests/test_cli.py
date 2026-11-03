"""doctor и auth: проверка связок и сценарий входа (без настоящего браузера)."""

from __future__ import annotations

import argparse
import contextlib
import io
import json
import os
import shutil
import stat
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import httpx

from llm_rotator import cli, doctor, load_registry
from llm_rotator.web.deepseek import auth as login
from llm_rotator.web.store import ConversationStore

PROVIDERS = """
[providers.api]
provider_name = "api"
base_url = "https://api.test/v1"
models_url = "https://api.test/v1/models"

[providers.dsw]
provider_name = "dsw"
api_style = "deepseek_web"
base_url = "https://chat.test"
models_url = ""
auth = "web_session"
[providers.dsw.web.models]
"w" = {}
"""
PROXIES = """
[proxies.direct]
proxy_name = "direct"
proxy_address = ""
[proxies.dead]
proxy_name = "dead"
proxy_address = "127.0.0.1:1"
"""
KEYS = """
[[keys]]
key_name = "API_OK"
provider = "api"
proxies = ["direct"]
[[keys]]
key_name = "API_BAD"
provider = "api"
proxies = ["direct"]
[[keys]]
key_name = "WEB_OK"
provider = "dsw"
key_file = "accounts/ok.json"
proxies = ["direct"]
[[keys]]
key_name = "WEB_DEAD"
provider = "dsw"
key_file = "accounts/dead.json"
proxies = ["direct"]
[[keys]]
key_name = "WEB_UNREACHABLE"
provider = "dsw"
key_file = "accounts/ok.json"
proxies = ["dead"]
"""


def envelope(biz_data=None):
    return {
        "code": 0,
        "msg": "",
        "data": {"biz_code": 0, "biz_msg": "", "biz_data": biz_data},
    }


class Base(unittest.TestCase):
    def setUp(self):
        self.dir = Path(tempfile.mkdtemp())
        self.addCleanup(shutil.rmtree, self.dir, ignore_errors=True)
        models = self.dir / "models"
        models.mkdir()
        (models / "providers.toml").write_text(PROVIDERS)
        (models / "proxies.toml").write_text(PROXIES)
        (models / "keys.toml").write_text(KEYS)
        (models / "models.toml").write_text(
            '[[models]]\nmodel_name = "a"\nproviders = ["api"]\n'
            '[[models]]\nmodel_name = "w"\nproviders = ["dsw"]\n'
        )
        (self.dir / ".env").write_text("API_OK=good\nAPI_BAD=bad\n")
        (self.dir / "accounts").mkdir()
        (self.dir / "accounts" / "ok.json").write_text(
            json.dumps({"token": "tok-good-123"})
        )
        (self.dir / "accounts" / "dead.json").write_text(
            json.dumps({"token": "tok-dead-456"})
        )


def handler(request: httpx.Request) -> httpx.Response:
    auth = request.headers.get("authorization", "")
    if request.url.host == "api.test":
        return httpx.Response(200 if auth == "Bearer good" else 401, json={"data": []})
    if "tok-good-123" in auth:
        return httpx.Response(200, json=envelope({"chat_sessions": []}))
    return httpx.Response(200, json={"code": 40003, "msg": "INVALID_TOKEN"})


class DoctorTests(Base):
    def run_doctor(self, store=None):
        registry = load_registry(self.dir)
        http = httpx.Client(transport=httpx.MockTransport(handler))
        self.addCleanup(http.close)
        return doctor.run(registry, store, http_for=lambda proxy: http), registry

    def test_every_binding_is_reported(self):
        rows, _ = self.run_doctor()
        by = {r.binding: r for r in rows}
        self.assertTrue(by["API_OK@direct"].ok)
        self.assertFalse(by["API_BAD@direct"].ok)
        self.assertIn("не принят", by["API_BAD@direct"].detail)
        self.assertTrue(by["WEB_OK@direct"].ok)
        self.assertFalse(by["WEB_DEAD@direct"].ok)
        self.assertFalse(by["WEB_UNREACHABLE@dead"].ok)
        self.assertIn("не отвечает", by["WEB_UNREACHABLE@dead"].detail)

    def test_dead_web_account_is_marked_and_gets_the_relogin_command(self):
        store = ConversationStore()
        self.addCleanup(store.close)
        rows, _ = self.run_doctor(store)
        dead = next(r for r in rows if r.binding.startswith("WEB_DEAD"))
        self.assertEqual(dead.hint, "llm-rotator auth deepseek WEB_DEAD")
        self.assertEqual(store.account_health("WEB_DEAD"), "needs_auth")
        self.assertEqual(store.account_health("WEB_OK"), "ok")

    def test_render_lists_direct_web_accounts_and_hides_secrets(self):
        rows, registry = self.run_doctor()
        text = doctor.render(rows, doctor.direct_web_accounts(registry))
        self.assertIn("FAIL", text)
        self.assertIn("WEB_OK@direct", text.split("прямом IP")[-1])
        for secret in (
            "tok-good-123",
            "tok-dead-456",
            "Bearer",
            "ok.json",
            "dead.json",
        ):
            self.assertNotIn(secret, text)

    def test_cli_doctor_exit_code_and_output(self):
        real = httpx.Client
        out = io.StringIO()
        with (
            contextlib.redirect_stdout(out),
            mock.patch.object(
                doctor.httpx,
                "Client",
                side_effect=lambda **kw: real(transport=httpx.MockTransport(handler)),
            ),
        ):
            code = cli.main(["doctor", "--base-dir", str(self.dir)])
        self.assertEqual(code, 1)
        self.assertIn("auth deepseek WEB_DEAD", out.getvalue())
        self.assertTrue((self.dir / "state" / "rotator.db").exists())


class AuthCommandTests(Base):
    def args(self, key):
        return argparse.Namespace(
            site="deepseek", key_name=key, base_dir=self.dir, headless=False
        )

    def run_auth(self, key, session, chrome="/usr/bin/chromium", verify=0):
        out, err = io.StringIO(), io.StringIO()
        with (
            contextlib.redirect_stdout(out),
            contextlib.redirect_stderr(err),
            mock.patch.object(login, "capture", return_value=session) as capture,
            mock.patch.object(cli, "_verify", return_value=verify),
        ):
            code = cli._auth(self.args(key), prompt=lambda _: "", chrome=chrome)
        return code, out.getvalue(), err.getvalue(), capture

    def test_new_account_file_is_created_with_0600_and_proxy_is_passed(self):
        (self.dir / "models" / "keys.toml").write_text(
            KEYS + '[[keys]]\nkey_name = "NEW"\nprovider = "dsw"\n'
            'key_file = "accounts/new.json"\nproxies = ["direct"]\n'
        )
        session = login.SessionData(token="TOK", cookie="c=1", cookies_count=1)
        code, out, err, capture = self.run_auth("NEW", session)
        self.assertEqual(code, 0, err)
        path = self.dir / "accounts" / "new.json"
        self.assertEqual(json.loads(path.read_text())["token"], "TOK")
        self.assertEqual(stat.S_IMODE(os.stat(path).st_mode), 0o600)
        self.assertEqual(capture.call_args.kwargs["proxy"].name, "direct")
        self.assertIn("прямого IP", out)
        self.assertNotIn("TOK", out + err)  # значение токена не печатается

    def test_incomplete_login_leaves_old_file(self):
        before = (self.dir / "accounts" / "ok.json").read_text()
        code, _, err, _ = self.run_auth(
            "WEB_OK", login.SessionData(token="", cookie="")
        )
        self.assertEqual(code, 2)
        self.assertIn("не тронут", err)
        self.assertEqual((self.dir / "accounts" / "ok.json").read_text(), before)

    def test_unknown_key_and_api_key_are_rejected(self):
        code, _, err, _ = self.run_auth("NOPE", login.SessionData("t", "c"))
        self.assertEqual(code, 1)
        self.assertIn("API_OK", err)
        code, _, err, _ = self.run_auth("API_OK", login.SessionData("t", "c"))
        self.assertEqual(code, 1)
        self.assertIn("не DeepSeek", err)

    def test_missing_browser_is_a_clear_error(self):
        with mock.patch.object(login, "find_chrome", return_value=None):
            out, err = io.StringIO(), io.StringIO()
            with contextlib.redirect_stdout(out), contextlib.redirect_stderr(err):
                code = cli._auth(self.args("WEB_OK"), prompt=lambda _: "", chrome=None)
        self.assertEqual(code, 1)
        self.assertIn("CHROME_PATH", err.getvalue())

    def test_proxy_with_password_is_refused_before_launch(self):
        (self.dir / "models" / "proxies.toml").write_text(
            PROXIES
            + '[proxies.auth]\nproxy_name = "auth"\nproxy_address = "http://u:p@h:1"\n'
        )
        (self.dir / "models" / "keys.toml").write_text(
            KEYS + '[[keys]]\nkey_name = "PW"\nprovider = "dsw"\n'
            'key_file = "accounts/pw.json"\nproxies = ["auth"]\n'
        )
        err = io.StringIO()
        with contextlib.redirect_stderr(err), contextlib.redirect_stdout(io.StringIO()):
            code = cli._auth(self.args("PW"), prompt=lambda _: "", chrome="/bin/true")
        self.assertEqual(code, 2)
        self.assertIn("логин и пароль", err.getvalue())
        self.assertNotIn(":p@", err.getvalue())


if __name__ == "__main__":
    unittest.main()
