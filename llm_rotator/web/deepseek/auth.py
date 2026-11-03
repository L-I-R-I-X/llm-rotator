"""Вход в аккаунт DeepSeek Web: захват сессии из Chrome/Chromium через CDP.

Порт `scripts/deepseek_chrome_auth.js`. Отличия от оригинала:

- Chrome запускается с **прокси связки** (`--proxy-server`): вход должен идти с
  того же IP, через который аккаунт потом работает;
- профиль у каждого ключа свой (`state/chrome/<key_name>`), порт отладки
  свободный, а не фиксированный;
- файл сессии пишется атомарно и с правами 600; неполный вход старый файл
  не затирает.

Нужен extra `[auth]` (websockets). Значения токена и cookie не печатаются.
"""

from __future__ import annotations

import json
import os
import re
import shutil
import socket
import subprocess
import threading
import time
import urllib.request
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

from ...registry import Proxy

LOGIN_URL = "https://chat.deepseek.com/"
DEFAULT_WASM = "https://fe-static.deepseek.com/chat/static/sha3_wasm_bg.7b9ca65ddd.wasm"
COOKIE_DOMAIN = r"deepseek\.com$"
CHROME_NAMES = (
    "chromium",
    "google-chrome",
    "google-chrome-stable",
    "chromium-browser",
    "chrome",
)
TOKEN_KEYS = ("userToken", "token", "auth_token", "access_token", "accessToken")


class AuthError(Exception):
    """Понятная пользователю причина, почему вход не удался."""


@dataclass(frozen=True)
class SessionData:
    token: str
    cookie: str
    hif_dliq: str = ""
    hif_leim: str = ""
    wasm_url: str = DEFAULT_WASM
    base_url: str = "https://chat.deepseek.com"
    cookies_count: int = 0
    href: str = ""

    @property
    def complete(self) -> bool:
        return bool(self.token and self.cookie)

    def as_file(self) -> dict:
        """Содержимое файла сессии (формат `account.load_account`)."""
        return {
            "token": self.token,
            "cookie": self.cookie,
            "hif_dliq": self.hif_dliq,
            "hif_leim": self.hif_leim,
            "wasmUrl": self.wasm_url,
            "baseUrl": self.base_url,
        }


# --- Chrome ----------------------------------------------------------------------


def find_chrome(env: dict[str, str] | None = None) -> str | None:
    env = os.environ if env is None else env
    explicit = env.get("CHROME_PATH")
    if explicit:
        return explicit if os.path.exists(explicit) else None
    for name in CHROME_NAMES:
        found = shutil.which(name)
        if found:
            return found
    return None


def chrome_proxy_arg(proxy: Proxy) -> str | None:
    """Значение `--proxy-server` для прокси связки. `None` — идти напрямую.

    Chrome не принимает логин и пароль во флаге, поэтому такой прокси —
    понятная ошибка, а не молчаливый вход с чужого адреса.
    """
    if proxy.is_direct:
        return None
    address = proxy.httpx_proxy or ""
    if "@" in address:
        raise AuthError(
            f"прокси {proxy.name!r} требует логин и пароль, а Chrome не принимает их "
            "во флаге --proxy-server. Войдите через прокси без пароля (по IP-доступу) "
            "или импортируйте файл сессии вручную."
        )
    scheme, _, rest = address.partition("://")
    scheme = {"socks5h": "socks5"}.get(scheme, scheme)
    return f"{scheme}://{rest}"


def chrome_args(
    chrome: str,
    profile: Path,
    port: int,
    proxy_arg: str | None,
    url: str,
    headless: bool,
) -> list[str]:
    args = [
        chrome,
        f"--user-data-dir={profile}",
        f"--remote-debugging-port={port}",
        "--remote-debugging-address=127.0.0.1",
        "--password-store=basic",
        "--use-mock-keychain",
        "--disable-sync",
        "--disable-extensions",
        "--no-first-run",
        "--no-default-browser-check",
        "--disable-infobars",
    ]
    if proxy_arg:
        args.append(f"--proxy-server={proxy_arg}")
    if headless:
        args.append("--headless=new")
    args.append(url)
    return args


def free_port() -> int:
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        return sock.getsockname()[1]


def _json(url: str, timeout: float = 2.0):
    with urllib.request.urlopen(url, timeout=timeout) as response:  # noqa: S310 — loopback
        return json.load(response)


def wait_devtools(port: int, timeout: float = 20.0) -> dict:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        try:
            return _json(f"http://127.0.0.1:{port}/json/version")
        except OSError:
            time.sleep(0.25)
    raise AuthError(
        "Chrome не поднял DevTools — закройте окно этого профиля и повторите"
    )


def page_target(port: int, timeout: float = 10.0) -> dict:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        targets = _json(f"http://127.0.0.1:{port}/json")
        pages = [
            t
            for t in targets
            if t.get("type") == "page" and t.get("webSocketDebuggerUrl")
        ]
        if pages:
            return pages[0]
        time.sleep(0.25)
    raise AuthError("не нашёл вкладку Chrome")


class CDP:
    """Минимальный клиент Chrome DevTools Protocol на `websockets` (sync)."""

    def __init__(self, ws_url: str) -> None:
        try:
            from websockets.sync.client import connect
        except ImportError:
            raise AuthError(
                "нужен пакет websockets: pip install 'llm-rotator[auth]'"
            ) from None
        self._ws = connect(ws_url, max_size=None, open_timeout=10)
        self._id = 0
        self._lock = threading.Lock()
        self._pending: dict[int, dict] = {}
        self._cond = threading.Condition()
        self.events: list[dict] = []
        self._closed = False
        self._reader = threading.Thread(target=self._read, daemon=True)
        self._reader.start()

    def _read(self) -> None:
        try:
            for raw in self._ws:
                message = json.loads(raw)
                with self._cond:
                    if "id" in message:
                        self._pending[message["id"]] = message
                    elif "method" in message:
                        self.events.append(message)
                        del self.events[:-2000]
                    self._cond.notify_all()
        except Exception:  # noqa: BLE001 — сокет закрыт
            pass
        finally:
            with self._cond:
                self._closed = True
                self._cond.notify_all()

    def send(
        self, method: str, params: dict | None = None, timeout: float = 15.0
    ) -> dict:
        with self._lock:
            self._id += 1
            call = self._id
        self._ws.send(
            json.dumps({"id": call, "method": method, "params": params or {}})
        )
        deadline = time.monotonic() + timeout
        with self._cond:
            while call not in self._pending:
                left = deadline - time.monotonic()
                if self._closed or left <= 0:
                    raise AuthError(f"CDP {method}: нет ответа")
                self._cond.wait(left)
            message = self._pending.pop(call)
        if "error" in message:
            raise AuthError(
                f"CDP {method}: {message['error'].get('message', 'ошибка')}"
            )
        return message.get("result", {})

    def close(self) -> None:
        try:
            self._ws.close()
        except Exception:  # noqa: BLE001
            pass


# --- чтение сессии ---------------------------------------------------------------

_PAGE_JS = """(() => {
  const out = {href: location.href, localStorage: {}, sessionStorage: {}, resources: []};
  for (let i = 0; i < localStorage.length; i++) { const k = localStorage.key(i); out.localStorage[k] = localStorage.getItem(k); }
  for (let i = 0; i < sessionStorage.length; i++) { const k = sessionStorage.key(i); out.sessionStorage[k] = sessionStorage.getItem(k); }
  out.resources = performance.getEntriesByType('resource').map(r => r.name).filter(n => /wasm|chat\\/completion|pow|chat_session/.test(n)).slice(-100);
  return out;
})()"""


def normalize_token(raw: object) -> str:
    if not raw:
        return ""
    try:
        parsed = json.loads(str(raw))
    except ValueError:
        return str(raw).strip()
    if isinstance(parsed, dict):
        for key in ("value", "token", "access_token", "accessToken"):
            if parsed.get(key):
                return str(parsed[key])
        return ""
    return str(parsed).strip() if isinstance(parsed, str) else str(raw).strip()


def token_from_storage(stores: list[dict]) -> str:
    for store in stores:
        for key in TOKEN_KEYS:
            token = normalize_token(store.get(key))
            if token:
                return token
    for store in stores:
        for key, value in store.items():
            if re.search("token", key, re.IGNORECASE):
                token = normalize_token(value)
                if token:
                    return token
    return ""


def session_from(
    page: dict,
    cookies: list[dict],
    events: list[dict],
    cookie_domain: str = COOKIE_DOMAIN,
    base_url: str = "https://chat.deepseek.com",
) -> SessionData:
    token = token_from_storage(
        [page.get("localStorage") or {}, page.get("sessionStorage") or {}]
    )
    mine = [c for c in cookies if re.search(cookie_domain, c.get("domain", ""))]
    cookie = "; ".join(f"{c['name']}={c['value']}" for c in mine)
    dliq = leim = ""
    for event in events:
        params = event.get("params", {})
        headers = (
            params.get("headers") or (params.get("request") or {}).get("headers") or {}
        )
        for name, value in headers.items():
            lower = name.lower()
            if lower == "x-hif-dliq":
                dliq = str(value)
            elif lower == "x-hif-leim":
                leim = str(value)
            elif (
                lower == "authorization"
                and not token
                and re.match(r"Bearer\s+", str(value), re.I)
            ):
                token = re.sub(r"^Bearer\s+", "", str(value), flags=re.I)
    wasm = next(
        (u for u in page.get("resources") or [] if re.search(r"sha3.*\.wasm", u)),
        DEFAULT_WASM,
    )
    return SessionData(
        token=token,
        cookie=cookie,
        hif_dliq=dliq,
        hif_leim=leim,
        wasm_url=wasm,
        base_url=base_url,
        cookies_count=len(mine),
        href=str(page.get("href", "")),
    )


def read_session(
    cdp: CDP,
    cookie_domain: str = COOKIE_DOMAIN,
    base_url: str = "https://chat.deepseek.com",
) -> SessionData:
    result = cdp.send(
        "Runtime.evaluate", {"expression": _PAGE_JS, "returnByValue": True}
    )
    page = result.get("result", {}).get("value") or {}
    cookies = cdp.send("Network.getAllCookies").get("cookies", [])
    return session_from(page, cookies, list(cdp.events), cookie_domain, base_url)


def write_session(path: Path, session: SessionData) -> None:
    """Файл сессии: атомарно, права 600. Неполный вход — `AuthError`, файл цел."""
    if not session.complete:
        raise AuthError(
            "вход не завершён (нет токена или cookie), файл сессии не тронут"
        )
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    fd = os.open(tmp, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600)
    with os.fdopen(fd, "w", encoding="utf-8") as handle:
        json.dump(session.as_file(), handle, ensure_ascii=False, indent=2)
    os.chmod(tmp, 0o600)
    os.replace(tmp, path)


# --- сценарий --------------------------------------------------------------------


def capture(
    *,
    chrome: str,
    profile: Path,
    proxy: Proxy,
    wait_for_user: Callable[[], None],
    url: str = LOGIN_URL,
    cookie_domain: str = COOKIE_DOMAIN,
    base_url: str = "https://chat.deepseek.com",
    headless: bool = False,
    settle_seconds: float = 10.0,
) -> SessionData:
    """Открыть Chrome через прокси связки, дождаться входа, забрать сессию."""
    proxy_arg = chrome_proxy_arg(proxy)
    profile.mkdir(parents=True, exist_ok=True)
    port = free_port()
    process = subprocess.Popen(  # noqa: S603 — путь к браузеру выбран пользователем
        chrome_args(chrome, profile, port, proxy_arg, url, headless),
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
    cdp: CDP | None = None
    try:
        wait_devtools(port)
        cdp = CDP(page_target(port)["webSocketDebuggerUrl"])
        cdp.send("Runtime.enable")
        cdp.send("Network.enable")
        wait_for_user()
        deadline = time.monotonic() + settle_seconds
        session = read_session(cdp, cookie_domain, base_url)
        while not session.complete and time.monotonic() < deadline:
            time.sleep(0.5)
            session = read_session(cdp, cookie_domain, base_url)
        return session
    finally:
        if cdp is not None:
            try:
                cdp.send("Browser.close", timeout=3)
            except AuthError:
                pass
            cdp.close()
        try:
            process.wait(timeout=5)
        except subprocess.TimeoutExpired:
            process.terminate()
