"""Командная строка: `llm-rotator serve | auth | accounts | doctor`."""

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
    auth.add_argument(
        "name",
        nargs="?",
        help="имя аккаунта: есть в keys.toml — перелогин, нет — новый аккаунт "
        "accounts/<имя>.json. Без имени: цикл входов, имя спросят после каждого",
    )
    auth.add_argument("--base-dir", type=Path, default=Path.cwd())
    auth.add_argument(
        "--provider", help="провайдер нового аккаунта, если DeepSeek Web не один"
    )
    auth.add_argument(
        "--headless", action="store_true", help="для тестов; входить придётся вслепую"
    )
    accounts = sub.add_parser("accounts", help="файлы аккаунтов в accounts/")
    accounts_sub = accounts.add_subparsers(dest="accounts_command", required=True)
    sync = accounts_sub.add_parser(
        "sync", help="завести в keys.toml блоки для файлов из accounts/"
    )
    sync.add_argument("--base-dir", type=Path, default=Path.cwd())
    sync.add_argument("--provider", help="если DeepSeek Web-провайдеров несколько")
    sync.add_argument(
        "--dry-run", action="store_true", help="только показать, что будет добавлено"
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
    if args.command == "accounts":
        return _accounts_sync(args)
    return 2


def _auth(args: argparse.Namespace, prompt=input, chrome: str | None = None) -> int:
    from . import accounts
    from .registry import ConfigError, load_registry
    from .web.deepseek import auth as login

    base = args.base_dir.resolve()
    try:
        registry = load_registry(base, check_key_files=False)
    except ConfigError as exc:
        print(f"Ошибка конфигурации: {exc}", file=sys.stderr)
        return 1
    chrome = chrome or login.find_chrome()
    if not chrome:
        print(
            "Не нашёл Chrome/Chromium. Установите браузер или укажите путь: "
            "CHROME_PATH=/usr/bin/chromium llm-rotator auth …",
            file=sys.stderr,
        )
        return 1
    name = getattr(args, "name", None)
    if name:
        try:
            name = accounts.normalize_name(name)
        except ValueError as exc:
            print(exc, file=sys.stderr)
            return 1
        if name in registry.keys:
            return _relogin(registry, name, args, chrome, prompt)
    try:
        provider = accounts.pick_provider(registry, getattr(args, "provider", None))
    except ValueError as exc:
        print(exc, file=sys.stderr)
        return 1
    if name:
        return _new_account(registry, provider, name, args, chrome, prompt)
    return _enrollment_loop(registry, provider, args, chrome, prompt)


def _show_proxy(proxy, count: int = 1) -> None:
    if count > 1:
        print(f"У ключа несколько прокси; вход идёт через первый: {proxy.name}")
    if proxy.is_direct:
        print(
            "Прокси у связки нет: вход идёт с прямого IP. Когда назначите прокси, "
            "может понадобиться войти заново через него."
        )
    else:
        print(f"Вход через прокси {proxy.name}")


def _capture(chrome: str, profile: Path, proxy, headless: bool, prompt):
    """Открыть браузер, дождаться входа, показать итог. Токены не печатаются."""
    from .web.deepseek import auth as login

    def wait() -> None:
        print(
            "\nChrome открыт. Войдите в DeepSeek в ЭТОМ окне и отправьте любое сообщение (ok)."
        )
        prompt("Когда войдёте — нажмите ENTER здесь: ")

    session = login.capture(
        chrome=chrome,
        profile=profile,
        proxy=proxy,
        wait_for_user=wait,
        headless=headless,
    )
    print(f"страница: {session.href or '?'}")
    print(
        f"токен: {'есть, ' + str(len(session.token)) + ' символов' if session.token else 'НЕТ'}"
    )
    print(
        f"cookie: {'есть, ' + str(session.cookies_count) + ' шт.' if session.cookie else 'НЕТ'}"
    )
    return session


def _relogin(registry, name: str, args, chrome: str, prompt) -> int:
    from .web.deepseek import auth as login

    key = registry.keys[name]
    provider = registry.provider(key.provider)
    if provider.api_style != "deepseek_web":
        print(f"{name}: провайдер {provider.name!r} — не DeepSeek Web", file=sys.stderr)
        return 1
    proxy = registry.proxy(key.proxies[0])
    _show_proxy(proxy, len(key.proxies))
    profile = args.base_dir.resolve() / "state" / "chrome" / name
    try:
        session = _capture(chrome, profile, proxy, args.headless, prompt)
        login.write_session(Path(key.key_file), session)
    except login.AuthError as exc:
        print(f"Не сохранено: {exc}", file=sys.stderr)
        return 2
    print(f"Сохранено: {key.key_file} (права 600)")
    return _verify(registry, name)


def _new_account(registry, provider: str, name: str, args, chrome: str, prompt) -> int:
    """Аккаунт с заданным именем: вход, файл accounts/<имя>.json и блок в keys.toml."""
    from .web.deepseek import auth as login

    proxy = registry.proxy("direct")
    _show_proxy(proxy)
    profile = args.base_dir.resolve() / "state" / "chrome" / name
    try:
        session = _capture(chrome, profile, proxy, args.headless, prompt)
        return _store_new(registry, provider, name, session)
    except login.AuthError as exc:
        print(f"Не сохранено: {exc}", file=sys.stderr)
        return 2


def _store_new(registry, provider: str, name: str, session) -> int:
    """Записать файл сессии и завести ключ. Сначала файл: без него ключ бесполезен."""
    from . import accounts
    from .registry import load_registry
    from .web.deepseek import auth as login

    path = accounts.account_path(registry.base_dir, name)
    login.write_session(path, session)
    accounts.append_key(registry, name, provider, path)
    print(f"Сохранено: {path} (права 600), ключ {name} добавлен в keys.toml")
    fresh = load_registry(registry.base_dir, check_key_files=False)
    return _verify(fresh, name)


def _ask_name(registry, taken: set[str], prompt) -> str:
    """Имя нового аккаунта: Enter — DS_n, повтор при занятом или кривом имени."""
    from . import accounts

    auto = accounts.next_free_name(taken, registry.base_dir)
    while True:
        raw = prompt(f"Имя аккаунта (Enter — {auto}): ").strip()
        if not raw:
            return auto
        try:
            name = accounts.normalize_name(raw)
        except ValueError as exc:
            print(exc)
            continue
        if name in taken or accounts.account_path(registry.base_dir, name).exists():
            print(f"{name}: такой аккаунт уже есть, введите другое имя")
            continue
        return name


def _enrollment_loop(registry, provider: str, args, chrome: str, prompt) -> int:
    """Вход в несколько новых аккаунтов подряд. Ctrl+C — закончить."""
    import shutil

    from .registry import load_registry
    from .web.deepseek import auth as login

    base = args.base_dir.resolve()
    proxy = registry.proxy("direct")
    _show_proxy(proxy)
    print("Цикл входов в новые аккаунты. Ctrl+C — закончить.")
    saved: list[str] = []
    failed = 0
    while True:
        # Свежий профиль на каждый круг: иначе браузер сам войдёт в прошлый аккаунт.
        profile = base / "state" / "chrome" / f"_new-{os.getpid()}"
        shutil.rmtree(profile, ignore_errors=True)
        try:
            try:
                session = _capture(chrome, profile, proxy, args.headless, prompt)
                if not session.complete:  # имя спрашиваем только после удачного входа
                    raise login.AuthError(
                        "вход не завершён (нет токена или cookie), файл сессии не тронут"
                    )
                taken = set(registry.keys) | set(saved)
                name = _ask_name(registry, taken, prompt)
                registry = load_registry(base, check_key_files=False)
                _store_new(registry, provider, name, session)
                saved.append(name)
                registry = load_registry(base, check_key_files=False)
            except login.AuthError as exc:
                failed += 1
                print(f"Не сохранено: {exc}", file=sys.stderr)
            finally:
                shutil.rmtree(profile, ignore_errors=True)
            prompt("\nEnter — войти в ещё один аккаунт, Ctrl+C — закончить: ")
        except (KeyboardInterrupt, EOFError):
            print()
            break
    if saved:
        print(f"Добавлено аккаунтов: {len(saved)} ({', '.join(saved)}).")
        print("Чтобы сервис их увидел: systemctl --user restart llm-rotator")
    else:
        print("Ни один аккаунт не добавлен.")
    return 0 if saved or not failed else 2


def _accounts_sync(args: argparse.Namespace) -> int:
    from . import accounts
    from .registry import ConfigError, load_registry

    try:
        registry = load_registry(args.base_dir.resolve(), check_key_files=False)
        provider = accounts.pick_provider(registry, args.provider)
    except (ConfigError, ValueError) as exc:
        print(exc, file=sys.stderr)
        return 1
    pending = accounts.unregistered(registry)
    if not pending:
        print("Все файлы из accounts/ уже есть в keys.toml.")
        return 0
    if args.dry_run:
        for path in pending:
            print(f"будет добавлен: {path.name}")
        return 0
    added, skipped = accounts.sync(registry, provider, pending)
    for name, path in added:
        print(f"добавлен: {name}  <- {path.name}")
    for path, reason in skipped:
        print(f"пропущен: {path.name} ({reason})", file=sys.stderr)
    if added:
        print("Чтобы сервис их увидел: systemctl --user restart llm-rotator")
    return 0 if not skipped else 2


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
    from . import accounts

    orphans = accounts.unregistered(service.runner.registry)
    if orphans:
        print(
            "ВНИМАНИЕ: в accounts/ есть файлы без записи в keys.toml "
            f"({', '.join(p.name for p in orphans)}): llm-rotator accounts sync",
            file=sys.stderr,
        )
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
