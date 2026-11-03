"""Эмуляция tool calling для веб-чатов, у которых нет API инструментов.

Порт из FreeDeepseekAPI (`server.js:956-1580`): модели в промпте объясняют
формат запроса инструмента, а в ответе терпимо разбирают его — JSON, DSML/XML-
обёртки, чинят типичные ошибки JSON (обратные слэши Windows, сырые переводы
строк). Исполняемым считается только явный конверт: «голый» пример JSON,
вызов внутри чужого блока кода или во вложенном JSON не исполняется.
"""

from __future__ import annotations

import json
import re
import uuid
from collections.abc import Mapping, Sequence
from dataclasses import dataclass

TOOL_CALL_FORMAT = '{"tool_call":{"name":"<function_name>","arguments":{...}}}'

MAX_TOOL_MARKUP_CHARS = 256 * 1024
MAX_TOOL_ARGUMENT_CHARS = 128 * 1024
MAX_TOOL_JSON_CANDIDATES = 32
MAX_DSML_PARAMETERS = 128
MAX_DSML_STRUCTURAL_TAGS = MAX_DSML_PARAMETERS * 2 + 16
MAX_DSML_TAG_CHARS = 2048

_A = re.ASCII
_I = re.IGNORECASE


@dataclass(frozen=True, slots=True)
class ParsedCall:
    name: str
    arguments: dict


def new_call_id() -> str:
    return f"call_{uuid.uuid4().hex[:24]}"


# --- промпт -----------------------------------------------------------------

_ANNOTATION_KEYS = frozenset({"description", "examples", "$comment", "title"})
_MAP_KEYS = frozenset(
    {"properties", "patternProperties", "$defs", "definitions", "dependentSchemas"}
)
_ARRAY_KEYS = frozenset({"allOf", "anyOf", "oneOf", "prefixItems"})
_SINGLE_KEYS = frozenset(
    {
        "additionalItems",
        "additionalProperties",
        "contains",
        "contentSchema",
        "else",
        "if",
        "items",
        "not",
        "propertyNames",
        "then",
        "unevaluatedItems",
        "unevaluatedProperties",
    }
)


def compact_tool_schema(value: object) -> object:
    """Схема без описаний и примеров: на проверку аргументов они не влияют.

    Обходятся только ключевые слова, значения которых сами схемы; литералы под
    const/enum/default остаются как есть, даже если в них есть поля
    «description» или «title».
    """
    if isinstance(value, list):
        return [compact_tool_schema(v) for v in value]
    if not isinstance(value, dict):
        return value
    out: dict = {}
    for key, child in value.items():
        if key in _ANNOTATION_KEYS:
            continue
        if key in _MAP_KEYS and isinstance(child, dict):
            out[key] = {n: compact_tool_schema(s) for n, s in child.items()}
        elif key in _ARRAY_KEYS and isinstance(child, list):
            out[key] = [compact_tool_schema(s) for s in child]
        elif key in _SINGLE_KEYS:
            out[key] = (
                [compact_tool_schema(s) for s in child]
                if isinstance(child, list)
                else compact_tool_schema(child)
            )
        elif key == "dependencies" and isinstance(child, dict):
            out[key] = {
                n: d if isinstance(d, list) else compact_tool_schema(d)
                for n, d in child.items()
            }
        else:
            out[key] = child
    return out


def _functions(tools: object) -> list[Mapping]:
    out = []
    for tool in tools or ():
        if isinstance(tool, Mapping):
            function = tool.get("function")
            if tool.get("type", "function") == "function" and isinstance(
                function, Mapping
            ):
                out.append(function)
    return out


def tool_names(tools: object) -> list[str]:
    return [str(f["name"]) for f in _functions(tools) if f.get("name")]


def _dumps(value: object) -> str:
    return json.dumps(value, ensure_ascii=False, separators=(",", ":"))


def format_tool_definitions(tools: object, max_prompt_chars: int) -> str:
    """Инструкция для модели: формат запроса и список функций."""
    functions = _functions(tools)
    if not functions:
        return ""
    raw_chars = sum(len(_dumps(f.get("parameters") or {})) for f in functions)
    compact = raw_chars > int(max_prompt_chars * 0.4)
    text = "\n\n--- TOOL REQUEST SYSTEM ---\n"
    text += (
        "You are an AI that ONLY REASONS and REQUESTS tool executions. "
        "You do NOT run any commands yourself.\n"
        "When you need data from the user's machine or need to change something "
        "there, REQUEST exactly one tool call using exactly this JSON format:\n"
    )
    text += f"{TOOL_CALL_FORMAT}\n\n"
    text += (
        "Your response will be sent to a local gateway, which executes the "
        "command and sends the output back in the next message.\n\nRULES:\n"
        "1. You ONLY output the tool request — you never run anything yourself\n"
        "2. Do NOT simulate, guess, or fabricate command output — wait for the "
        "actual result\n"
        "3. The tool runs on the user's local machine — NOT on the chat site\n"
        "4. After the tool executes, the result will be sent to you as a new "
        "message starting with [Tool Result]\n"
        "5. A tool request contains nothing but that JSON: no explanation before "
        "or after it, no Markdown code fence, one tool per response\n"
        "6. Arguments must be valid JSON: escape backslashes (C:\\\\Users\\\\me), "
        "double quotes and newlines (\\n) inside strings. Keep arguments compact.\n"
        "7. When no tool is needed (for example the task is finished), answer "
        "normally in plain text without any tool_call JSON\n\n"
        "Available functions:\n"
    )
    for function in functions:
        text += f"\n## {function.get('name')}\n"
        description = " ".join(str(function.get("description") or "").split())
        if len(description) > 500:
            description = description[:497] + "..."
        text += f"{description}\n"
        parameters = function.get("parameters")
        if parameters:
            shown = compact_tool_schema(parameters) if compact else parameters
            text += f"Parameters: {_dumps(shown)}\n"
    text += "\n--- END TOOL REQUEST SYSTEM ---\n"
    text += (
        f"\nREMEMBER: Request a tool only with {TOOL_CALL_FORMAT}. "
        "Never simulate results."
    )
    return text


def format_tool_reminder(tools: object) -> str:
    """Короткое напоминание для дельта-ходов: полная инструкция лежит в начале чата."""
    names = tool_names(tools)
    if not names:
        return ""
    listing = ", ".join(names)
    if len(listing) > 1500:
        listing = listing[:1497] + "..."
    return (
        f"[Tool reminder] Available tools: {listing}. To use one, reply with ONLY "
        f"{TOOL_CALL_FORMAT} (valid JSON, one tool, nothing else). Otherwise "
        "answer normally. Never invent tool results."
    )


def format_retry_prompt() -> str:
    """Уточнение в том же чате, если разметка вызова оказалась битой."""
    return (
        "Your previous reply contained a malformed tool call, so it could not be "
        f"executed. Reply again with ONLY {TOOL_CALL_FORMAT} (valid JSON, one "
        "tool, no explanation, no Markdown code fence), or answer in plain text "
        "if no tool is needed."
    )


# --- разбор JSON ------------------------------------------------------------


def extract_balanced_json_at(text: str, start: int) -> str | None:
    if start >= len(text) or text[start] != "{":
        return None
    depth = 0
    in_string = False
    escape = False
    for i in range(start, len(text)):
        ch = text[i]
        if escape:
            escape = False
            continue
        if ch == "\\" and in_string:
            escape = True
            continue
        if ch == '"':
            in_string = not in_string
            continue
        if not in_string:
            if ch == "{":
                depth += 1
            elif ch == "}":
                depth -= 1
                if depth == 0:
                    return text[start : i + 1]
    return None


def extract_balanced_json_objects(
    text: str, max_objects: int = MAX_TOOL_JSON_CANDIDATES, accept=None
) -> list[str]:
    """Объекты верхнего уровня за один линейный проход."""
    objects: list[str] = []
    start = -1
    depth = 0
    in_string = False
    escape = False
    for i, ch in enumerate(text):
        if start == -1:
            if ch == "{":
                start, depth, in_string, escape = i, 1, False, False
            continue
        if escape:
            escape = False
            continue
        if ch == "\\" and in_string:
            escape = True
            continue
        if ch == '"':
            in_string = not in_string
            continue
        if in_string:
            continue
        if ch == "{":
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0:
                obj = text[start : i + 1]
                if accept is None or accept(obj):
                    objects.append(obj)
                    if len(objects) >= max_objects:
                        return objects
                start = -1
    return objects


_VALID_ESCAPE = re.compile(r'^(?:["\\/bfnrt]|u[0-9a-fA-F]{4})')
_SAFE_ESCAPE = re.compile(r'^(?:["\\]|u[0-9a-fA-F]{4})')
_ENVELOPE_KEY = re.compile(r"""["'](?:tool_call|tool_calls|function_call)["']\s*:""")
_DRIVE_PATH = re.compile(r"^[A-Za-z]:\\(?!\\)")


def _repair_json_string_body(body: str) -> str:
    # «Сырой» текст вроде пути Windows: одиночные обратные слэши буквальные,
    # "C:\new\tmp" остаётся как есть, а не превращается в перевод строки и tab.
    raw = bool(_DRIVE_PATH.match(body))
    k = 0
    while not raw and k < len(body):
        if body[k] == "\\":
            following = body[k + 1 : k + 6]
            if not _VALID_ESCAPE.match(following) or following[:1] in ("b", "f"):
                raw = True
            k += 1
        k += 1
    out: list[str] = []
    k = 0
    while k < len(body):
        c = body[k]
        if c == "\\":
            if raw and not _SAFE_ESCAPE.match(body[k + 1 : k + 6]):
                out.append("\\\\")
            else:
                out.append(c + (body[k + 1] if k + 1 < len(body) else ""))
                k += 1
            k += 1
            continue
        code = ord(c)
        if code < 0x20:
            out.append({"\n": "\\n", "\r": "\\r", "\t": "\\t"}.get(c, f"\\u{code:04x}"))
        else:
            out.append(c)
        k += 1
    return "".join(out)


def repair_json_text(raw: str) -> str:
    """Чинит две ошибки в остальном верного JSON: сырые обратные слэши и
    сырые переводы строк внутри строк. Структура вне строк не меняется."""
    text = str(raw or "")
    out: list[str] = []
    changed = False
    i = 0
    while i < len(text):
        if text[i] != '"':
            out.append(text[i])
            i += 1
            continue
        j = i + 1
        while j < len(text) and text[j] != '"':
            j += 2 if text[j] == "\\" else 1
        if j >= len(text):
            out.append(text[i:])
            break
        body = text[i + 1 : j]
        fixed = _repair_json_string_body(body)
        changed = changed or fixed != body
        out.append(f'"{fixed}"')
        i = j + 1
    return "".join(out) if changed else text


def _reject_constant(name: str):
    raise ValueError(name)


def _loads(raw: str):
    return json.loads(raw, parse_constant=_reject_constant)


def parse_json_lenient(raw: str):
    """json.loads с починкой. Даже валидный JSON чинится, если строка — явный
    путь Windows («C:\\files\\bin» парсится, но в перевод страницы и backspace)."""
    error: Exception | None = None
    parsed = None
    try:
        parsed = _loads(raw)
    except ValueError as exc:
        error = exc
    repaired = repair_json_text(raw)
    if repaired != raw:
        try:
            return _loads(repaired)
        except ValueError:
            pass
    if error is not None:
        raise error
    return parsed


_NAME = re.compile(r"^[A-Za-z0-9_][A-Za-z0-9_.:-]{0,127}$")


def build_tool_call(name: object, args: object = None) -> ParsedCall | None:
    tool_name = name.strip() if isinstance(name, str) else ""
    if not _NAME.match(tool_name):
        return None
    parsed = args
    if isinstance(parsed, str):
        if len(parsed) > MAX_TOOL_ARGUMENT_CHARS:
            return None
        try:
            parsed = parse_json_lenient(parsed)
        except ValueError:
            return None
    if parsed is None:
        parsed = {}
    if not isinstance(parsed, dict):
        return None
    try:
        serialized = json.dumps(parsed, ensure_ascii=False)
    except (TypeError, ValueError):
        return None
    if len(serialized) > MAX_TOOL_ARGUMENT_CHARS:
        return None
    return ParsedCall(tool_name, parsed)


def _first(*values):
    for value in values:
        if value is not None:
            return value
    return None


def coerce_tool_call_object(obj: object, allow_bare: bool = False) -> ParsedCall | None:
    if not isinstance(obj, dict):
        return None
    if "tool_call" in obj:
        candidate = obj["tool_call"]
    elif "function_call" in obj:
        candidate = obj["function_call"]
    elif "tool_calls" in obj:
        calls = obj["tool_calls"]
        if not isinstance(calls, list) or len(calls) != 1:
            return None
        candidate = calls[0]
    elif allow_bare:
        candidate = obj
    else:
        return None
    if not isinstance(candidate, dict):
        return None
    function = (
        candidate["function"]
        if isinstance(candidate.get("function"), dict)
        else candidate
    )
    return build_tool_call(
        _first(function.get("name"), candidate.get("name")),
        _first(
            function.get("arguments"),
            candidate.get("arguments"),
            candidate.get("input"),
            {},
        ),
    )


def _json_candidate(raw: str, allow_bare: bool = False) -> ParsedCall | None:
    if not raw:
        return None
    try:
        return coerce_tool_call_object(parse_json_lenient(raw), allow_bare)
    except ValueError:
        return None


# --- DSML / XML -------------------------------------------------------------


def _canonical_tag(raw_tag: str) -> str | None:
    token = (
        str(raw_tag or "")
        .strip()
        .replace("｜", "|")
        .translate(
            {
                0x201C: '"',
                0x201D: '"',
                0xFF02: '"',
                0x2018: "'",
                0x2019: "'",
                0xFF07: "'",
            }
        )
    )
    closing = False
    if token.startswith("/"):
        closing = True
        token = token[1:].strip()
    token = re.sub(r"^\|+\s*DSML\s*\|+\s*", "", token, flags=_I)
    if token.startswith("/"):
        closing = True
        token = token[1:].strip()
    token = re.sub(
        r"^DSML(?=(?:tool[\s_-]*calls|function[\s_-]*calls|invoke|parameter)\b)",
        "",
        token,
        flags=_I | _A,
    )
    if not closing and re.match(r"^name\s*=", token, _I):
        return f"<direct {token}>"
    semantic = re.match(
        r"^(?:(?:[A-Za-z_][\w.-]*):)?(tool[\s_-]*calls|function[\s_-]*calls|invoke|parameter)\b([\s\S]*)$",
        token,
        _I | _A,
    )
    if not semantic:
        return None
    local = re.sub(r"[\s_-]", "", semantic.group(1)).lower()
    name = "tool_calls" if local in ("toolcalls", "functioncalls") else local
    attrs = "" if closing else semantic.group(2)
    return f"<{'/' if closing else ''}{name}{attrs}>"


def _normalize_tags(text: str) -> str:
    text = str(text or "").replace("＜", "<").replace("＞", ">")
    return re.sub(
        r"<([^<>]{0,1024})>", lambda m: _canonical_tag(m.group(1)) or m.group(0), text
    )


def _decode_value(value: str) -> str:
    out = str(value or "")
    for pattern, repl in (
        (r"&quot;", '"'),
        (r"&apos;", "'"),
        (r"&lt;", "<"),
        (r"&gt;", ">"),
        (r"&amp;", "&"),
    ):
        out = re.sub(pattern, repl, out, flags=_I)
    return out


def _decode_parameter(value: str) -> str:
    raw = str(value or "")
    cdata = re.match(r"^<!\[CDATA\[([\s\S]*?)\]\]>$", raw.strip(), _I)
    return cdata.group(1) if cdata else _decode_value(raw)


def _attribute(attrs: str, attribute: str) -> str | None:
    match = re.search(
        rf"""\b{attribute}\s*=\s*(["'])([^"']+)\1""", str(attrs or ""), _I | _A
    )
    return match.group(2) if match else None


@dataclass(frozen=True, slots=True)
class _Tag:
    name: str
    attrs: str
    closing: bool
    self_closing: bool
    start: int
    end: int


_INVALID = object()


def _read_tag(text: str, start: int):
    if text[start] != "<":
        return None
    prefix = text[start + 1 : min(len(text), start + 40)].lstrip()
    if not re.match(r"^/?(?:tool_calls|invoke|parameter|direct)\b", prefix, _I | _A):
        return None
    quote = None
    end = -1
    for i in range(start + 1, min(len(text), start + MAX_DSML_TAG_CHARS + 1)):
        ch = text[i]
        if quote:
            if ch == quote:
                quote = None
            continue
        if ch in "\"'":
            quote = ch
            continue
        if ch == ">":
            end = i
            break
    if end == -1:
        return _INVALID
    token = text[start + 1 : end].strip()
    closing = False
    if token.startswith("/"):
        closing = True
        token = token[1:].strip()
    self_closing = False
    if not closing and token.endswith("/"):
        self_closing = True
        token = token[:-1].strip()
    match = re.match(
        r"^(tool_calls|invoke|parameter|direct)\b([\s\S]*)$", token, _I | _A
    )
    if not match:
        return None
    return _Tag(
        match.group(1).lower(),
        "" if closing else match.group(2),
        closing,
        self_closing,
        start,
        end + 1,
    )


def _scan_tags(text: str) -> list[_Tag] | None:
    tags: list[_Tag] = []
    value = str(text or "")
    i = 0
    while i < len(value):
        if value[i : i + 9].upper() == "<![CDATA[":
            end = value.find("]]>", i + 9)
            if end == -1:
                return None
            i = end + 3
            continue
        if value[i] != "<":
            i += 1
            continue
        tag = _read_tag(value, i)
        if tag is None:
            i += 1
            continue
        if tag is _INVALID:
            return None
        tags.append(tag)
        if len(tags) > MAX_DSML_STRUCTURAL_TAGS:
            return None
        i = tag.end
    return tags


def _parse_parameter(attrs: str, raw_body: str, args: dict, seen: set) -> bool:
    name = _attribute(attrs, "name")
    if not name or not _NAME.match(name) or name in seen:
        return False
    seen.add(name)
    string_mode = _attribute(attrs, "string")
    raw_value = _decode_parameter(raw_body)
    if len(raw_value) > MAX_TOOL_ARGUMENT_CHARS:
        return False
    value: object = raw_value
    if string_mode and string_mode.lower() == "false":
        try:
            value = parse_json_lenient(raw_value.strip())
        except ValueError:
            return False
    args[name] = value
    return True


def _parse_invoke(name: str | None, body: str) -> ParsedCall | None:
    tags = _scan_tags(body)
    if tags is None:
        return None
    parameter_tags = [t for t in tags if t.name == "parameter"]
    if any(t.name != "parameter" for t in tags):
        return None
    args: dict = {}
    count = 0
    seen: set = set()
    cursor = 0
    for i in range(0, len(parameter_tags), 2):
        opening = parameter_tags[i]
        closing = parameter_tags[i + 1] if i + 1 < len(parameter_tags) else None
        if (
            opening.closing
            or opening.self_closing
            or closing is None
            or not closing.closing
        ):
            return None
        if body[cursor : opening.start].strip():
            return None
        count += 1
        if count > MAX_DSML_PARAMETERS:
            return None
        if not _parse_parameter(
            opening.attrs, body[opening.end : closing.start], args, seen
        ):
            return None
        cursor = closing.end
    if count > 0:
        if body[cursor:].strip():
            return None
        return build_tool_call(name, args)
    decoded = _decode_value(body).strip()
    if not decoded:
        return build_tool_call(name, {})
    objects = extract_balanced_json_objects(decoded, 2)
    if len(objects) != 1 or decoded != objects[0]:
        return None
    try:
        return build_tool_call(name, parse_json_lenient(objects[0]))
    except ValueError:
        return None


def _tool_call_scope(normalized: str) -> str | None:
    tags = _scan_tags(normalized)
    if tags is None:
        return None
    wrappers = [t for t in tags if t.name == "tool_calls"]
    openings = [t for t in wrappers if not t.closing]
    closings = [t for t in wrappers if t.closing]
    if openings:
        if len(openings) != 1 or openings[0].self_closing or not closings:
            return None
        opening, closing = openings[0], closings[-1]
        if (
            any(t.closing and t.start < opening.end for t in wrappers)
            or closing.start < opening.end
        ):
            return None
        if any(
            t.name != "tool_calls"
            and (t.start < opening.end or t.start >= closing.start)
            for t in tags
        ):
            return None
        return normalized[opening.end : closing.start]
    # Узкая починка: нет открывающей обёртки, но закрывающая есть.
    if closings:
        closing = closings[-1]
        invokes = [
            t
            for t in tags
            if t.name == "invoke" and not t.closing and t.start < closing.start
        ]
        if len(invokes) == 1 and not invokes[0].self_closing:
            if any(
                t.name != "tool_calls"
                and (t.start < invokes[0].start or t.start >= closing.start)
                for t in tags
            ):
                return None
            return normalized[invokes[0].start : closing.start]
    return None


def _parse_dsml(text: str) -> ParsedCall | None:
    if len(str(text or "")) > MAX_TOOL_MARKUP_CHARS:
        return None
    normalized = _normalize_tags(text)
    scope = _tool_call_scope(normalized)
    if scope is None:
        return None
    tags = _scan_tags(scope)
    if not tags:
        return None
    first = tags[0]
    if scope[: first.start].strip():
        return None
    if first.name == "invoke" and not first.closing and not first.self_closing:
        invokes = [t for t in tags if t.name == "invoke"]
        if (
            len(invokes) != 2
            or invokes[0] is not first
            or invokes[1].closing is not True
        ):
            return None
        closing = invokes[1]
        if scope[closing.end :].strip():
            return None
        if any(t.name in ("tool_calls", "direct") for t in tags):
            return None
        parsed = _parse_invoke(
            _attribute(first.attrs, "name"), scope[first.end : closing.start]
        )
        if parsed:
            return parsed
    if first.name == "direct" and not first.closing and not first.self_closing:
        if any(
            i > 0 and t.name in ("direct", "invoke", "tool_calls")
            for i, t in enumerate(tags)
        ):
            return None
        parsed = _parse_invoke(_attribute(first.attrs, "name"), scope[first.end :])
        if parsed:
            return parsed
    return None


# --- верхний уровень --------------------------------------------------------

_FENCE = re.compile(r"(`{3,})([\w+.-]*)[^\S\r\n]*\r?\n?([\s\S]*?)\1(?!`)")
_JSON_LANG = re.compile(r"^(?:json[c5]?|tool_?calls?|tool|text|plaintext|txt)$", _I)
_DSML_HINT = re.compile(
    r"[|｜]+\s*DSML\s*[|｜]+|[<＜]\s*/?\s*(?:DSML)?(?:[\w.-]+:)?(?:tool[\s_-]*calls|function[\s_-]*calls|invoke)\b",
    _I | _A,
)
_MARKUP_HINT = re.compile(
    r"""TOOL_CALL:\s*[\w-]+|<\s*tool_call\b|[|｜]+\s*DSML\s*[|｜]+|[<＜]\s*/?\s*(?:DSML)?(?:[\w.-]+:)?(?:tool[\s_-]*calls|function[\s_-]*calls|invoke)\b|["'](?:tool_call|tool_calls|function_call)["']\s*:""",
    _I | _A,
)


def _is_json_lang(language: str) -> bool:
    return not language or bool(_JSON_LANG.match(language))


def strip_foreign_code_fences(text: str) -> str:
    """Блоки кода на других языках — исходники, вызовы внутри них не исполняются."""
    return _FENCE.sub(
        lambda m: m.group(0) if _is_json_lang(m.group(2)) else "\n", str(text or "")
    )


def looks_like_tool_call_markup(raw: str) -> bool:
    return bool(_MARKUP_HINT.search(strip_foreign_code_fences(raw)))


def parse_tool_call(text: str) -> ParsedCall | None:
    """Вызов инструмента из ответа модели; `None` — это обычный текст."""
    if not text or not isinstance(text, str) or len(text) > MAX_TOOL_MARKUP_CHARS:
        return None
    scan = strip_foreign_code_fences(text)

    if _DSML_HINT.search(scan):
        return _parse_dsml(scan)

    xml = re.search(r"<tool_call[^>]*>([\s\S]*?)</tool_call>", scan, _I)
    if xml:
        inner = xml.group(1).strip()
        call = _json_candidate(inner, allow_bare=True)
        if not call:
            start = inner.find("{")
            obj = None if start == -1 else extract_balanced_json_at(inner, start)
            if obj and obj != inner:
                call = _json_candidate(obj, allow_bare=True)
        if call:
            return call

    for fence in _FENCE.finditer(text):
        if not _is_json_lang(fence.group(2)):
            continue
        body = fence.group(3).strip()
        if not body.startswith("{"):
            continue
        call = _json_candidate(body)
        if call:
            return call

    legacy = re.search(r"TOOL_CALL:\s*([\w-]+)\s*", scan, _I | _A)
    if legacy:
        after = scan[legacy.end() :]
        brace = after.find("{")
        if brace != -1:
            raw_json = extract_balanced_json_at(after, brace)
            if raw_json:
                try:
                    call = build_tool_call(
                        legacy.group(1), parse_json_lenient(raw_json)
                    )
                except ValueError:
                    call = None
                if call:
                    return call

    for raw_json in extract_balanced_json_objects(
        scan, MAX_TOOL_JSON_CANDIDATES, lambda obj: bool(_ENVELOPE_KEY.search(obj))
    ):
        call = _json_candidate(raw_json)
        if call:
            return call
    return None


def known_call(call: ParsedCall | None, tools: Sequence | object) -> ParsedCall | None:
    """Вызов только известного инструмента; выдуманное имя — не вызов."""
    if call is None:
        return None
    return call if call.name in tool_names(tools) else None
