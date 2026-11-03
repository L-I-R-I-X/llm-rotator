"""HTTP-клиент DeepSeek Web: без состояния разговоров, одна связка = один клиент.

Создаётся на (аккаунт, `httpx.Client` через прокси связки). Любая ошибка —
`WebError`: HTTP-код сам по себе ничего не значит, потому что DeepSeek отдаёт
отказы с кодом 200 (`code 40003`, `biz_code != 0`, `event: hint` в SSE).
"""

from __future__ import annotations

import json
import re
import threading
import time
from collections.abc import Callable, Iterator, Sequence
from urllib.parse import quote

import httpx

from ...registry import WebModel, WebSettings
from .. import errors
from ..errors import WebError
from . import pow as pow_module
from .account import Account
from .sse import Event, iter_events

CLIENT_VERSION = "2.0.0"
USER_AGENT = (
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 "
    "(KHTML, like Gecko) Chrome/149.0.0.0 Safari/537.36"
)
COMPLETION_PATH = "/api/v0/chat/completion"
UPLOAD_PATH = "/api/v0/file/upload_file"

CONNECT_TIMEOUT = 10.0
FILE_TIMEOUT = 60.0
FILE_POLL = 1.0

_AUTH_RE = re.compile(r"token|authori[sz]|login|unauthori[sz]", re.IGNORECASE)
_FILE_READY = re.compile(r"^(?:SUCCESS|PARSED|DONE|COMPLETED?|FINISHED|READY)$")
_FILE_FAILED = re.compile(r"FAIL|ERROR|REJECT|UNSUPPORT|INVALID|EMPTY")

PowSolver = Callable[[dict, str, httpx.Client], int]


def _retry_after(headers: httpx.Headers) -> float | None:
    try:
        return max(0.0, float(headers.get("retry-after", "")))
    except ValueError:
        return None


class _Watchdog:
    """Закрывает ответ, если в потоке слишком долго тишина.

    До первого SSE-события лимит `first`, дальше — `idle`. Сам httpx умеет
    только один таймаут чтения на весь запрос, а нам нужны два.
    """

    def __init__(self, response: httpx.Response, first: float, idle: float) -> None:
        self._response = response
        self._idle = idle
        self._timer: threading.Timer | None = None
        self._lock = threading.Lock()
        self.fired = False
        self._arm(first)

    def _arm(self, seconds: float) -> None:
        with self._lock:
            if self._timer:
                self._timer.cancel()
            self._timer = threading.Timer(seconds, self._fire)
            self._timer.daemon = True
            self._timer.start()

    def _fire(self) -> None:
        self.fired = True
        self._response.close()

    def event(self) -> None:
        self._arm(self._idle)

    def stop(self) -> None:
        with self._lock:
            if self._timer:
                self._timer.cancel()


class DeepSeekClient:
    def __init__(
        self,
        account: Account,
        http: httpx.Client,
        settings: WebSettings,
        base_url: str,
        pow_solver: PowSolver = pow_module.solve,
        sleep: Callable[[float], None] = time.sleep,
    ) -> None:
        self._account = account
        self._http = http
        self._settings = settings
        self._base = base_url.rstrip("/")
        self._solve = pow_solver
        self._sleep = sleep
        self._timeout = httpx.Timeout(
            connect=CONNECT_TIMEOUT,
            read=settings.first_event_timeout,
            write=30.0,
            pool=CONNECT_TIMEOUT,
        )

    def __repr__(self) -> str:
        return f"DeepSeekClient({self._base})"

    # --- запросы ------------------------------------------------------------

    def _headers(self, extra: dict[str, str] | None = None, json_body: bool = True):
        account = self._account
        headers = {
            "User-Agent": USER_AGENT,
            "x-client-platform": "web",
            "x-client-version": CLIENT_VERSION,
            "x-client-locale": "ru",
            "x-client-timezone-offset": str(time.localtime().tm_gmtoff or 0),
            "x-app-version": CLIENT_VERSION,
            "Authorization": f"Bearer {account.token}",
            "x-hif-dliq": account.hif_dliq,
            "x-hif-leim": account.hif_leim,
            "Origin": self._base,
            "Referer": f"{self._base}/",
            "Cookie": account.cookie,
        }
        if json_body:
            headers["Content-Type"] = "application/json"
        headers.update(extra or {})
        return headers

    def _url(self, path: str) -> str:
        return f"{self._base}{path}"

    def _send(self, method: str, path: str, **kwargs) -> httpx.Response:
        try:
            return self._http.request(
                method, self._url(path), timeout=self._timeout, **kwargs
            )
        except httpx.TimeoutException:
            raise WebError("timeout", f"{path}: нет ответа") from None
        except httpx.HTTPError as exc:
            raise WebError("network", f"{path}: {type(exc).__name__}") from None

    def _call(self, method: str, path: str, body: dict | None = None, **kw) -> dict:
        headers = self._headers(kw.pop("headers", None), json_body=body is not None)
        content = json.dumps(body) if body is not None else None
        response = self._send(method, path, headers=headers, content=content, **kw)
        return self._parse(response)

    def _parse(self, response: httpx.Response) -> dict:
        """Тело → dict, либо `WebError` по коду, телу или конверту `biz_code`."""
        text = response.text
        try:
            body = json.loads(text) if text.strip() else None
        except ValueError:
            body = None
        status = response.status_code
        if status >= 400:
            raise self._http_error(status, text, body, _retry_after(response.headers))
        if not isinstance(body, dict):
            raise WebError("upstream", "ответ не JSON", status=status)
        self._check_envelope(body, status)
        return body

    @staticmethod
    def _http_error(status: int, text: str, body: object, after: float | None):
        detail = f"HTTP {status}: {text[:200]}"
        if status == 401:
            return WebError("auth", detail, status=status)
        if status == 403:
            # С JSON — отказ аккаунту; без — Cloudflare/WAF закрыл адрес.
            kind = "auth" if isinstance(body, dict) else "blocked"
            return WebError(kind, detail, status=status)
        if status == 429:
            return WebError("rate_limit", detail, retry_after=after, status=status)
        if status == 413:
            return WebError("too_large", detail, status=status)
        if status == 400:
            kind = errors.classify_text(text) or "bad_request"
            return WebError(kind, detail, status=status)
        if status >= 500:
            return WebError("upstream", detail, retry_after=after, status=status)
        return WebError("bad_request", detail, status=status)

    @staticmethod
    def _check_envelope(body: dict, status: int) -> None:
        """Отказы внутри HTTP 200: `code` и `data.biz_code`."""
        code = body.get("code")
        if code not in (0, None):
            msg = str(body.get("msg") or "")
            if code == 40003 or _AUTH_RE.search(msg):
                raise WebError("auth", f"code {code}: {msg}", status=status)
            kind = errors.classify_text(msg) or "upstream"
            raise WebError(kind, f"code {code}: {msg}", status=status)
        data = body.get("data")
        biz = data.get("biz_code") if isinstance(data, dict) else None
        if biz not in (0, None):
            msg = str(data.get("biz_msg") or "")
            if "invalid chat session" in msg.lower():
                raise WebError("chat_gone", msg, status=status)
            if _AUTH_RE.search(msg):
                raise WebError("auth", f"biz_code {biz}: {msg}", status=status)
            kind = errors.classify_text(msg) or "upstream"
            raise WebError(kind, f"biz_code {biz}: {msg}", status=status)

    # --- PoW ----------------------------------------------------------------

    def _pow_header(self, target_path: str) -> str:
        body = self._call(
            "POST", "/api/v0/chat/create_pow_challenge", {"target_path": target_path}
        )
        challenge = ((body.get("data") or {}).get("biz_data") or {}).get("challenge")
        if not isinstance(challenge, dict):
            raise WebError("upstream", "в ответе PoW нет challenge (капча/сессия?)")
        try:
            answer = self._solve(challenge, self._account.wasm_url, self._http)
        except (KeyError, TypeError, ValueError) as exc:
            raise WebError(
                "upstream", f"челлендж PoW разобрать нельзя: {exc}"
            ) from None
        return pow_module.header_value(challenge, answer, target_path)

    # --- методы API ---------------------------------------------------------

    def create_chat(self) -> str:
        body = self._call("POST", "/api/v0/chat_session/create", {})
        biz = (body.get("data") or {}).get("biz_data") or {}
        session = biz.get("chat_session")
        chat_id = (session or {}).get("id") or biz.get("id")
        if not isinstance(chat_id, str) or not chat_id:
            raise WebError("upstream", "создание чата не вернуло id")
        return chat_id

    def delete_chat(self, chat_id: str) -> bool:
        """Удалить чат. `False` — его уже нет (это не ошибка)."""
        try:
            self._call(
                "POST", "/api/v0/chat_session/delete", {"chat_session_id": chat_id}
            )
        except WebError as exc:
            if exc.kind == "chat_gone":
                return False
            raise
        return True

    def ping(self) -> bool:
        """Дешёвая проверка токена; при мёртвом токене — `WebError("auth")`."""
        self._call("GET", "/api/v0/chat_session/fetch_page?count=2")
        return True

    def complete(
        self,
        chat_id: str,
        parent_id: int | str | None,
        prompt: str,
        ref_file_ids: Sequence[str] = (),
        flags: WebModel = WebModel(),
    ) -> Iterator[Event]:
        """Ответ потоком событий. Ошибки поднимаются при чтении итератора."""
        pow_value = self._pow_header(COMPLETION_PATH)
        body = {
            "chat_session_id": chat_id,
            "parent_message_id": parent_id,
            "prompt": prompt,
            "ref_file_ids": list(ref_file_ids),
            "thinking_enabled": flags.thinking,
            "search_enabled": flags.search,
            "action": None,
            "preempt": False,
            "model_type": "default",
        }
        headers = self._headers({"X-DS-PoW-Response": pow_value})
        s = self._settings
        timeout = httpx.Timeout(
            connect=CONNECT_TIMEOUT,
            read=max(s.first_event_timeout, s.idle_timeout),
            write=30.0,
            pool=CONNECT_TIMEOUT,
        )
        try:
            with self._http.stream(
                "POST",
                self._url(COMPLETION_PATH),
                headers=headers,
                content=json.dumps(body),
                timeout=timeout,
            ) as response:
                if not self._is_sse(response):
                    response.read()
                    self._parse(response)  # конверт с ошибкой при коде 200
                    raise WebError("upstream", "completion вернул не SSE")
                dog = _Watchdog(response, s.first_event_timeout, s.idle_timeout)
                try:
                    yield from iter_events(self._chunks(response, dog), dog.event)
                finally:
                    dog.stop()
        except httpx.TimeoutException:
            raise WebError("timeout", "completion: нет данных") from None
        except httpx.HTTPError as exc:
            raise WebError("network", f"completion: {type(exc).__name__}") from None

    @staticmethod
    def _is_sse(response: httpx.Response) -> bool:
        return response.status_code == 200 and "event-stream" in response.headers.get(
            "content-type", ""
        )

    @staticmethod
    def _chunks(response: httpx.Response, dog: _Watchdog) -> Iterator[bytes]:
        try:
            yield from response.iter_bytes()
        except httpx.HTTPError:
            if not dog.fired:
                raise
        if dog.fired:
            raise WebError("timeout", "в потоке слишком долгая тишина")

    def upload_file(
        self, name: str, data: bytes, mime: str = "application/octet-stream"
    ) -> str:
        """Загрузить файл и дождаться разбора; вернуть file_id для ref_file_ids."""
        pow_value = self._pow_header(UPLOAD_PATH)
        headers = self._headers(
            {"X-DS-PoW-Response": pow_value, "x-file-size": str(len(data))},
            json_body=False,
        )
        response = self._send(
            "POST", UPLOAD_PATH, headers=headers, files={"file": (name, data, mime)}
        )
        body = self._parse(response)
        biz = (body.get("data") or {}).get("biz_data") or {}
        file_id = (
            biz.get("id") or biz.get("file_id") or (biz.get("file") or {}).get("id")
        )
        if not isinstance(file_id, str) or not file_id:
            raise WebError("upstream", "загрузка файла не вернула id")
        status = str(biz.get("status") or "").upper()
        deadline = time.monotonic() + FILE_TIMEOUT
        while not _FILE_READY.match(status):
            if _FILE_FAILED.search(status):
                raise WebError("bad_request", f"файл не разобран (статус {status})")
            if time.monotonic() > deadline:
                raise WebError("timeout", "файл не разобран за отведённое время")
            self._sleep(FILE_POLL)
            polled = self._call(
                "GET", f"/api/v0/file/fetch_files?file_ids={quote(file_id)}"
            )
            status = _file_status(polled, file_id)
        return file_id


def _file_status(body: dict, file_id: str) -> str:
    """Статус файла; ответ бывает и списком, и словарём по id."""
    biz = (body.get("data") or {}).get("biz_data")
    if isinstance(biz, dict) and isinstance(biz.get("files"), list):
        entries = biz["files"]
    elif isinstance(biz, list):
        entries = biz
    elif isinstance(biz, dict) and isinstance(biz.get(file_id), dict):
        entries = [biz[file_id]]
    elif isinstance(biz, dict) and (biz.get("id") or biz.get("file_id")):
        entries = [biz]
    else:
        entries = []
    entries = [e for e in entries if isinstance(e, dict)]
    entry = next(
        (e for e in entries if file_id in (e.get("id"), e.get("file_id"))),
        entries[0] if len(entries) == 1 else None,
    )
    return str((entry or {}).get("status") or "").upper()
