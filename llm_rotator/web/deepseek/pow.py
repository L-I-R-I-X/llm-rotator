"""Proof-of-work для DeepSeek Web через WASM (`wasmtime`, extra `[web]`).

Порт `lib/pow.js`. Модуль WASM неизменен для данного URL, поэтому
компилируется один раз (кэш потокобезопасный), а на каждое решение создаётся
свежий экземпляр с чистой памятью. WASM качается тем же `httpx.Client`, что и
вся связка, то есть через её прокси: иначе адрес утёк бы.
"""

from __future__ import annotations

import base64
import json
import math
import struct
import threading

import httpx

from ..errors import WebError

WASM_TIMEOUT = 15.0

_lock = threading.Lock()
_engine = None
_modules: dict[str, object] = {}


def _load_module(wasm_url: str, http: httpx.Client):
    try:
        import wasmtime
    except ImportError:
        raise WebError(
            "bad_request", "для DeepSeek нужен пакет wasmtime: pip install '.[web]'"
        ) from None
    global _engine
    if not wasm_url:
        raise WebError("bad_request", "в файле сессии нет wasmUrl")
    with _lock:
        module = _modules.get(wasm_url)
        if module is not None:
            return module
        try:
            response = http.get(wasm_url, timeout=WASM_TIMEOUT)
        except httpx.TimeoutException:
            raise WebError("timeout", "загрузка WASM для PoW") from None
        except httpx.HTTPError:
            raise WebError("network", "загрузка WASM для PoW") from None
        if response.status_code != 200:
            raise WebError(
                "upstream", f"WASM для PoW: HTTP {response.status_code}", status=502
            )
        if _engine is None:
            _engine = wasmtime.Engine()
        module = wasmtime.Module(_engine, response.content)
        _modules[wasm_url] = module  # неудачную загрузку не кэшируем
        return module


def _number(param, value):
    """Число в тип параметра WASM: JS приводит само, Python — нет."""
    return float(value) if "f" in str(param).lower() else int(value)


def solve(challenge: dict, wasm_url: str, http: httpx.Client) -> int:
    """Найти `answer` для челленджа. Нет решения — `WebError("upstream")`."""
    import wasmtime

    module = _load_module(wasm_url, http)
    store = wasmtime.Store(_engine)
    instance = wasmtime.Instance(store, module, [])
    exports = instance.exports(store)
    memory = exports["memory"]
    alloc = exports["__wbindgen_export_0"]
    stack = exports["__wbindgen_add_to_stack_pointer"]
    solve_fn = exports["wasm_solve"]

    challenge_bytes = str(challenge["challenge"]).encode()
    prefix_bytes = f"{challenge['salt']}_{challenge['expire_at']}_".encode()

    def put(data: bytes) -> int:
        pointer = alloc(store, len(data), 1) & 0xFFFFFFFF
        memory.write(store, data, pointer)
        return pointer

    c_ptr = put(challenge_bytes)
    p_ptr = put(prefix_bytes)
    sp = stack(store, -16)
    try:
        solve_fn(
            store,
            sp,
            c_ptr,
            len(challenge_bytes),
            p_ptr,
            len(prefix_bytes),
            _number(solve_fn.type(store).params[-1], challenge["difficulty"]),
        )
        raw = memory.read(store, sp, sp + 16)
        code = struct.unpack_from("<i", raw, 0)[0]
        answer = struct.unpack_from("<d", raw, 8)[0]
    finally:
        stack(store, 16)
    # code 0 — решения нет; nonce 0 — нормальный ответ.
    if code == 0 or not math.isfinite(answer) or answer < 0:
        raise WebError("upstream", "PoW не решился")
    return math.floor(answer)


def header_value(challenge: dict, answer: int, target_path: str) -> str:
    """Значение заголовка `X-DS-PoW-Response`: base64 от JSON."""
    payload = {
        "algorithm": challenge["algorithm"],
        "challenge": challenge["challenge"],
        "salt": challenge["salt"],
        "answer": answer,
        "signature": challenge["signature"],
        "target_path": target_path,
    }
    return base64.b64encode(
        json.dumps(payload, separators=(",", ":")).encode()
    ).decode()
