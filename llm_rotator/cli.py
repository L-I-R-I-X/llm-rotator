"""Командная строка: `llm-rotator serve`."""

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
    args = parser.parse_args(argv)

    logging.basicConfig(
        level=args.log_level.upper(),
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    if args.command == "serve":
        return _serve(args)
    return 2


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
