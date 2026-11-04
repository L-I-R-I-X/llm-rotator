"""Файлы веб-аккаунтов: имена, записи в keys.toml и синхронизация с папкой.

Аккаунт — это файл сессии `accounts/<имя>.json` и блок `[[keys]]` в keys.toml.
Блок можно написать руками, а можно не писать: `auth` добавляет его сам после
входа, а `accounts sync` заводит блоки для файлов, о которых keys.toml не знает.
Убрать аккаунт — удалить его блок (файл сессии при этом остаётся лежать).
"""

from __future__ import annotations

import json
import re
from collections.abc import Collection, Iterable
from pathlib import Path

from .registry import DIRECT_PROXY, KEYS_FILE, Registry

ACCOUNTS_DIR = "accounts"
AUTO_PREFIX = "DS_"
# Имя может быть и почтой: user@gmail.com. Без «/» и «..» файл не выйдет из папки.
NAME_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.@+-]{0,63}$")
WEB_STYLE = "deepseek_web"


def normalize_name(raw: str) -> str:
    """Имя аккаунта из ввода пользователя; `.json` на конце отбрасывается."""
    name = raw.strip()
    if name.lower().endswith(".json"):
        name = name[: -len(".json")]
    if not NAME_RE.match(name) or ".." in name:
        raise ValueError(
            f"недопустимое имя {raw!r}: буквы, цифры и _ . @ + -, "
            "первым символом буква или цифра, до 64 символов"
        )
    return name


def accounts_dir(base_dir: Path) -> Path:
    return Path(base_dir) / ACCOUNTS_DIR


def account_path(base_dir: Path, name: str) -> Path:
    return accounts_dir(base_dir) / f"{name}.json"


def keys_path(registry: Registry) -> Path:
    return Path(registry.base_dir) / KEYS_FILE


def next_free_name(taken: Collection[str], base_dir: Path) -> str:
    """Первое свободное `DS_n`: нет ни такого ключа, ни такого файла."""
    n = 1
    while True:
        name = f"{AUTO_PREFIX}{n}"
        if name not in taken and not account_path(base_dir, name).exists():
            return name
        n += 1


def pick_provider(registry: Registry, wanted: str | None = None) -> str:
    """Провайдер для нового аккаунта: единственный DeepSeek Web или `--provider`."""
    web = [p.name for p in registry.providers.values() if p.api_style == WEB_STYLE]
    if wanted:
        if wanted not in web:
            raise ValueError(
                f"провайдер {wanted!r} — не DeepSeek Web. Есть: {', '.join(web) or '-'}"
            )
        return wanted
    if len(web) == 1:
        return web[0]
    if not web:
        raise ValueError("в providers.toml нет провайдера с api_style = deepseek_web")
    raise ValueError(
        f"провайдеров DeepSeek Web несколько ({', '.join(web)}): укажи --provider"
    )


def key_block(
    name: str, provider: str, relative_file: str, proxy: str = DIRECT_PROXY
) -> str:
    return (
        "\n[[keys]]\n"
        f"key_name = {json.dumps(name)}\n"
        f"provider = {json.dumps(provider)}\n"
        f"key_file = {json.dumps(relative_file)}\n"
        f"proxies = [{json.dumps(proxy)}]\n"
        "max_concurrent = 1\n"
    )


def relative_file(base_dir: Path, path: Path) -> str:
    try:
        return path.resolve().relative_to(Path(base_dir).resolve()).as_posix()
    except ValueError:
        return str(path.resolve())


def append_key(
    registry: Registry, name: str, provider: str, path: Path, proxy: str = DIRECT_PROXY
) -> None:
    """Дописать блок аккаунта в конец keys.toml. Существующее не трогаем."""
    target = keys_path(registry)
    text = target.read_text(encoding="utf-8") if target.exists() else ""
    if text and not text.endswith("\n"):
        text += "\n"
    block = key_block(name, provider, relative_file(registry.base_dir, path), proxy)
    target.write_text(text + block, encoding="utf-8")


def registered_files(registry: Registry) -> set[Path]:
    return {Path(k.key_file).resolve() for k in registry.keys.values() if k.key_file}


def unregistered(registry: Registry) -> list[Path]:
    """Файлы сессий в accounts/, которых нет ни у одного ключа keys.toml."""
    folder = accounts_dir(registry.base_dir)
    if not folder.is_dir():
        return []
    known = registered_files(registry)
    return sorted(p for p in folder.glob("*.json") if p.resolve() not in known)


def sync(
    registry: Registry, provider: str, files: Iterable[Path] | None = None
) -> tuple[list[tuple[str, Path]], list[tuple[Path, str]]]:
    """Завести блоки для файлов без ключа: `(добавлено, пропущено с причиной)`."""
    added: list[tuple[str, Path]] = []
    skipped: list[tuple[Path, str]] = []
    taken = set(registry.keys)
    for path in sorted(unregistered(registry) if files is None else files):
        try:
            name = normalize_name(path.stem)
        except ValueError as exc:
            skipped.append((path, str(exc)))
            continue
        if name in taken:
            name = next_free_name(taken, registry.base_dir)
        append_key(registry, name, provider, path)
        taken.add(name)
        added.append((name, path))
    return added, skipped
