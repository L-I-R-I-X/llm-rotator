"""Вход в аккаунт: прокси Chrome, разбор сессии, запись файла, живой Chromium."""

from __future__ import annotations

import http.server
import json
import os
import shutil
import stat
import tempfile
import threading
import unittest
from pathlib import Path

from llm_rotator import Proxy
from llm_rotator.web.deepseek import auth as login

try:
    import websockets  # noqa: F401

    HAVE_WS = True
except ImportError:
    HAVE_WS = False
CHROME = login.find_chrome()


class ProxyArgTests(unittest.TestCase):
    def test_direct_means_no_flag(self):
        self.assertIsNone(login.chrome_proxy_arg(Proxy("direct", "")))

    def test_http_and_socks(self):
        self.assertEqual(
            login.chrome_proxy_arg(Proxy("p", "10.0.0.1:3128")), "http://10.0.0.1:3128"
        )
        self.assertEqual(
            login.chrome_proxy_arg(Proxy("p", "10.0.0.1:1080", proxy_type="socks5h")),
            "socks5://10.0.0.1:1080",
        )

    def test_credentials_are_refused_with_a_clear_error(self):
        with self.assertRaises(login.AuthError) as ctx:
            login.chrome_proxy_arg(Proxy("p", "http://user:pw@10.0.0.1:3128"))
        self.assertIn("логин и пароль", str(ctx.exception))
        self.assertNotIn("pw", str(ctx.exception))

    def test_args_carry_proxy_profile_and_port(self):
        args = login.chrome_args(
            "/bin/chrome", Path("/p"), 9999, "http://1.2.3.4:80", "https://x/", False
        )
        self.assertIn("--proxy-server=http://1.2.3.4:80", args)
        self.assertIn("--user-data-dir=/p", args)
        self.assertIn("--remote-debugging-port=9999", args)
        self.assertNotIn("--headless=new", args)
        self.assertNotIn(
            "--proxy-server",
            " ".join(login.chrome_args("c", Path("/p"), 1, None, "u", True)),
        )


class SessionParsingTests(unittest.TestCase):
    def test_token_from_json_wrapped_storage_and_cookie_filter(self):
        page = {
            "href": "https://chat.deepseek.com/a/chat/s/1",
            "localStorage": {
                "userToken": json.dumps({"value": "TOK", "__version": "0"})
            },
            "sessionStorage": {},
            "resources": ["https://x/sha3_wasm_bg.AAA.wasm"],
        }
        cookies = [
            {"name": "a", "value": "1", "domain": ".deepseek.com"},
            {"name": "b", "value": "2", "domain": "chat.deepseek.com"},
            {"name": "evil", "value": "3", "domain": "example.com"},
        ]
        events = [
            {"params": {"headers": {"X-Hif-Dliq": "D", "x-hif-leim": "L"}}},
            {"params": {"request": {"headers": {"x-other": "1"}}}},
        ]
        session = login.session_from(page, cookies, events)
        self.assertEqual(
            (session.token, session.cookie, session.hif_dliq, session.hif_leim),
            ("TOK", "a=1; b=2", "D", "L"),
        )
        self.assertEqual(session.wasm_url, "https://x/sha3_wasm_bg.AAA.wasm")
        self.assertEqual(session.cookies_count, 2)
        self.assertTrue(session.complete)

    def test_bearer_header_is_the_last_resort_for_token(self):
        events = [{"params": {"headers": {"Authorization": "Bearer FROMHDR"}}}]
        session = login.session_from({}, [], events)
        self.assertEqual(session.token, "FROMHDR")
        self.assertFalse(session.complete)  # cookie нет

    def test_token_key_fallback_by_name(self):
        page = {"localStorage": {"weird_token_key": "abc"}}
        self.assertEqual(login.session_from(page, [], []).token, "abc")

    def test_default_wasm_when_none_seen(self):
        self.assertEqual(login.session_from({}, [], []).wasm_url, login.DEFAULT_WASM)


class WriteTests(unittest.TestCase):
    def setUp(self):
        self.dir = Path(tempfile.mkdtemp())
        self.addCleanup(shutil.rmtree, self.dir, ignore_errors=True)

    def test_file_is_0600_and_loadable(self):
        from llm_rotator.web.deepseek.account import load_account

        path = self.dir / "sub" / "a.json"
        login.write_session(path, login.SessionData(token="T", cookie="c=1"))
        self.assertEqual(stat.S_IMODE(os.stat(path).st_mode), 0o600)
        self.assertEqual(load_account(path).token, "T")
        self.assertEqual(list(path.parent.glob("*.tmp")), [])

    def test_incomplete_login_keeps_the_old_file(self):
        path = self.dir / "a.json"
        path.write_text('{"token": "OLD"}')
        with self.assertRaises(login.AuthError):
            login.write_session(path, login.SessionData(token="", cookie=""))
        self.assertEqual(json.loads(path.read_text()), {"token": "OLD"})


PAGE = b"""<!doctype html><title>t</title><script>
localStorage.setItem('userToken', JSON.stringify({value: 'LIVETOKEN', __version: '0'}));
document.cookie = 'sid=abc; path=/';
// Requests keep flowing like on a live page: CDP attaches after start.
setInterval(() => fetch('/api/ping?' + Date.now(), {headers: {'x-hif-dliq': 'DLIQ', 'x-hif-leim': 'LEIM'}}), 300);
</script>"""


class FakeProxy(http.server.BaseHTTPRequestHandler):
    """HTTP-прокси: отдаёт страницу на любой абсолютный GET и запоминает хосты."""

    hosts: list[str] = []

    def do_GET(self):
        FakeProxy.hosts.append(self.path)
        body = PAGE if "/api/" not in self.path else b"{}"
        self.send_response(200)
        self.send_header(
            "Content-Type",
            "text/html" if "/api/" not in self.path else "application/json",
        )
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def log_message(self, *args):
        pass


@unittest.skipUnless(CHROME and HAVE_WS, "нужны Chromium и websockets")
class RealChromeTests(unittest.TestCase):
    def test_login_goes_through_the_proxy_and_session_is_captured(self):
        FakeProxy.hosts = []
        server = http.server.ThreadingHTTPServer(("127.0.0.1", 0), FakeProxy)
        threading.Thread(target=server.serve_forever, daemon=True).start()
        self.addCleanup(server.shutdown)
        profile = Path(tempfile.mkdtemp())
        self.addCleanup(shutil.rmtree, profile, ignore_errors=True)
        proxy = Proxy("fake", f"127.0.0.1:{server.server_port}")
        session = login.capture(
            chrome=CHROME,
            profile=profile,
            proxy=proxy,
            wait_for_user=lambda: __import__("time").sleep(2),
            url="http://auth.test/",
            cookie_domain=r"auth\.test$",
            base_url="http://auth.test",
            headless=True,
            settle_seconds=5,
        )
        # Страница с недоступного без прокси хоста пришла именно через него.
        self.assertTrue(any("auth.test" in h for h in FakeProxy.hosts), FakeProxy.hosts)
        self.assertEqual(session.token, "LIVETOKEN")
        self.assertIn("sid=abc", session.cookie)
        self.assertEqual((session.hif_dliq, session.hif_leim), ("DLIQ", "LEIM"))
        self.assertTrue(session.complete)


if __name__ == "__main__":
    unittest.main()
