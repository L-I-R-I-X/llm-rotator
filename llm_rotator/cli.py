"""Командная строка: `llm-rotator serve | auth | doctor`."""

from __future__ import annotations

import argparse
import logging
import os
import sys
from pathlib import Path

DEFAULT_PORT = 9660
PORT_VAR = "LLM_ROTATOR_PORT"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="llm-rotator")
    sub = parser.add_subparsers(dest="command", required=True)
    serve = sub.add_parser("serve", help="OpenAI-совместимый HTTP-сервис")
    serve.add_argument(
        "--base-dir", type=Path, default=Path.cwd(), help="каталог с models/ и .env"
    )
    serve.add_argument(
        "--host", default="127.0.0.1", help="по умолчанию только loopback"
    )
    serve.add_argument(
        "--port", type=int, default=int(os.environ.get(PORT_VAR, DEFAULT_PORT))
    )
    serve.add_argument("--log-level", default="info")
    auth = sub.add_parser("auth", help="войти в веб-аккаунт и сохранить сессию")
    auth.add_argument("site", choices=["deepseek"])
    auth.add_argument("key_name", help="key_name связки из keys.toml")
    auth.add_argument("--base-dir", type=Path, default=Path.cwd())
    auth.add_argument(
        "--headless", action="store_true", help="для тестов; входить придётся вслепую"
    )
    doctor = sub.add_parser("doctor", help="проверить прокси, токены и ключи")
    doctor.add_argument("--base-dir", type=Path, default=Path.cwd())
    args = parser.parse_args(argv)

    logging.basicConfig(
        level=getattr(args, "log_level", "warning").upper(),
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    if args.command == "serve":
        return _serve(args)
    if args.command == "auth":
        return _auth(args)
    if args.command == "doctor":
        return _doctor(args)
    return 2


def _auth(args: argparse.Namespace, prompt=input, chrome: str | None = None) -> int:
    from .registry import ConfigError, load_registry
    from .web.deepseek import auth as login

    try:
        registry = load_registry(args.base_dir.resolve(), check_key_files=False)
    except ConfigError as exc:
        print(f"Ошибка конфигурации: {exc}", file=sys.stderr)
        return 1
    key = registry.keys.get(args.key_name)
    if key is None:
        print(
            f"Ключа {args.key_name!r} нет в keys.toml. Есть: {', '.join(registry.keys) or '-'}",
            file=sys.stderr,
        )
        return 1
    provider = registry.provider(key.provider)
    if provider.api_style != "deepseek_web":
        print(
            f"{args.key_name}: провайдер {provider.name!r} — не DeepSeek Web",
            file=sys.stderr,
        )
        return 1
    chrome = chrome or login.find_chrome()
    if not chrome:
        print(
            "Не нашёл Chrome/Chromium. Установите браузер или укажите путь: "
            "CHROME_PATH=/usr/bin/chromium llm-rotator auth …",
            file=sys.stderr,
        )
        return 1
    proxy = registry.proxy(key.proxies[0])
    if len(key.proxies) > 1:
        print(f"У ключа несколько прокси; вход идёт через первый: {proxy.name}")
    if proxy.is_direct:
        print(
            "Прокси у связки нет: вход идёт с прямого IP. Когда назначите прокси, "
            "может понадобиться войти заново через него."
        )
    else:
        print(f"Вход через прокси {proxy.name}")
    profile = args.base_dir.resolve() / "state" / "chrome" / args.key_name

    def wait() -> None:
        print(
            "\nChrome открыт. Войдите в DeepSeek в ЭТОМ окне и отправьте любое сообщение (ok)."
        )
        prompt("Когда войдёте — нажмите ENTER здесь: ")

    try:
        session = login.capture(
            chrome=chrome,
            profile=profile,
            proxy=proxy,
            wait_for_user=wait,
            headless=args.headless,
        )
        path = Path(key.key_file)
        print(f"страница: {session.href or '?'}")
        print(
            f"токен: {'есть, ' + str(len(session.token)) + ' символов' if session.token else 'НЕТ'}"
        )
        print(
            f"cookie: {'есть, ' + str(session.cookies_count) + ' шт.' if session.cookie else 'НЕТ'}"
        )
        login.write_session(path, session)
    except login.AuthError as exc:
        print(f"Не сохранено: {exc}", file=sys.stderr)
        return 2
    print(f"Сохранено: {path} (права 600)")
    return _verify(registry, args.key_name)


def _verify(registry, key_name: str) -> int:
    from . import doctor

    rows = [r for r in doctor.run(registry) if r.binding.startswith(f"{key_name}@")]
    for row in rows:
        print(("токен рабочий" if row.ok else f"проверка не прошла: {row.detail}"))
    return 0 if rows and all(r.ok for r in rows) else 3


def _doctor(args: argparse.Namespace) -> int:
    from . import doctor
    from .registry import ConfigError, load_registry
    from .web.store import ConversationStore

    base = args.base_dir.resolve()
    try:
        registry = load_registry(base, check_key_files=False)
    except ConfigError as exc:
        print(f"Ошибка конфигурации: {exc}", file=sys.stderr)
        return 1
    db = base / "state" / "rotator.db"
    store = (
        ConversationStore(db)
        if any(p.web for p in registry.providers.values())
        else None
    )
    try:
        rows = doctor.run(registry, store)
    finally:
        if store is not None:
            store.close()
    print(doctor.render(rows, doctor.direct_web_accounts(registry)))
    return 0 if rows and all(r.ok for r in rows) else 1


def _serve(args: argparse.Namespace) -> int:
    try:
        import uvicorn

        from .registry import ConfigError
        from .service.app import Service, create_app
    except ImportError as exc:
        print(
            f"Для сервиса нужны extras: pip install 'llm-rotator[server]' ({exc})",
            file=sys.stderr,
        )
        return 1
    try:
        service = Service(args.base_dir.resolve())
    except ConfigError as exc:
        print(f"Ошибка конфигурации: {exc}", file=sys.stderr)
        return 1
    for warning in service.runner.registry.warnings:
        print(f"ВНИМАНИЕ: {warning}", file=sys.stderr)
    if (
        args.host not in ("127.0.0.1", "::1", "localhost")
        and not service.clients.requires_token
    ):
        print(
            "Сервис слушает не loopback, а clients.toml не настроен: запросы с "
            "других адресов получат 401. Заведите клиентов с токенами.",
            file=sys.stderr,
        )
    uvicorn.run(
        create_app(service), host=args.host, port=args.port, log_level=args.log_level
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
