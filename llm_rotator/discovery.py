"""Стартовый опрос списка моделей у провайдеров.

При старте скрипта ко всем провайдерам из providers.toml уходит запрос на
полный список доступных моделей. Результат кладётся в models/models.discovered.toml
с отметкой времени, и в течение TTL_CACHE_TTL повторно не запрашивается.

Режим best-effort: любой сбой только логируется и никогда не мешает старту.

Опрос идёт обычными связками (ключ, прокси) провайдера — теми же, что и
боевые запросы, см. model_registry.Binding. Никакого отдельного «служебного»
прокси для discovery не заводится: если для ключа настроен адрес, опрос идёт
через него, иначе через direct.

models.toml при этом не трогается никогда. Слияние выполняется только явной
командой `python model_discovery.py --merge` и устроено по имени модели:

* модели, которой в models.toml нет, добавляются с providers = [<провайдер>];
* уже описанные модели и их списки providers не меняются никогда;
* если ту же модель отдаёт и другой провайдер, это только подсказка в отчёте —
  молча дописывать второго провайдера нельзя, это тихо расширило бы маршрут
  и увеличило бы счёт.
"""

from __future__ import annotations

import argparse
import logging
import sys
import tomllib
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from pathlib import Path

from .gateway import Gateway, GatewayError
from .registry import (
    DISCOVERED_FILE,
    MODELS_FILE,
    ConfigError,
    Model,
    Registry,
    load_registry,
    resolve,
)

log = logging.getLogger("model_discovery")

TTL_CACHE_TTL = timedelta(hours=24)
DISCOVERY_READ_TIMEOUT = 30.0


@dataclass(frozen=True, slots=True)
class Discovered:
    """Результат опроса одного провайдера."""

    provider: str
    model_names: tuple[str, ...]
    error: str = ""

    @property
    def ok(self) -> bool:
        return not self.error


def now() -> datetime:
    return datetime.now(UTC)


def read_cache(path: Path = DISCOVERED_FILE) -> dict:
    path = resolve(path)
    if not path.exists():
        return {}
    try:
        with path.open("rb") as fh:
            return tomllib.load(fh)
    except (tomllib.TOMLDecodeError, OSError) as exc:
        log.warning("Кэш %s не читается (%s), будет перезаписан", path.name, exc)
        return {}


def cache_age(path: Path = DISCOVERED_FILE) -> timedelta | None:
    """Сколько прошло с момента последнего успешного опроса."""
    generated = read_cache(path).get("generated_at")
    if not isinstance(generated, str):
        return None
    try:
        return now() - datetime.fromisoformat(generated)
    except ValueError:
        log.warning("Некорректный generated_at в %s: %r", path.name, generated)
        return None


def cache_is_fresh(
    path: Path = DISCOVERED_FILE, ttl: timedelta = TTL_CACHE_TTL
) -> bool:
    age = cache_age(path)
    return age is not None and age < ttl


def poll_provider(gateway: Gateway, provider: str) -> Discovered:
    """Опросить одного провайдера. Ошибка возвращается в Discovered, не бросается."""
    try:
        names = gateway.list_models(provider, read_timeout=DISCOVERY_READ_TIMEOUT)
    except (GatewayError, ConfigError) as exc:
        return Discovered(provider=provider, model_names=(), error=str(exc))
    return Discovered(provider=provider, model_names=tuple(sorted(set(names))))


def poll_all(gateway: Gateway, registry: Registry) -> dict[str, Discovered]:
    """Опросить всех провайдеров параллельно."""
    names = list(registry.providers)
    if not names:
        return {}
    results: dict[str, Discovered] = {}
    with ThreadPoolExecutor(max_workers=len(names)) as pool:
        futures = {name: pool.submit(poll_provider, gateway, name) for name in names}
        for name, future in futures.items():
            try:
                results[name] = future.result()
            except Exception as exc:
                log.exception("Непредвиденный сбой при опросе %s", name)
                results[name] = Discovered(
                    provider=name, model_names=(), error=f"{type(exc).__name__}: {exc}"
                )
    return results


def toml_str(value: str) -> str:
    escaped = value.replace("\\", "\\\\").replace('"', '\\"')
    return f'"{escaped}"'


def toml_list(values: tuple[str, ...] | list[str]) -> str:
    return "[" + ", ".join(toml_str(v) for v in values) + "]"


def render_cache(results: dict[str, Discovered]) -> str:
    """Собрать models.discovered.toml. Провайдеры с ошибкой в файл не пишутся."""
    lines = [
        "# models/models.discovered.toml — кэш последнего опроса списка моделей.",
        "# Генерируется автоматически, руками не правится.",
        "# Служит только для удобства заполнения models/models.toml.",
        "",
        f'generated_at = "{now().isoformat()}"',
        "",
    ]
    for name, result in sorted(results.items()):
        if not result.ok:
            continue
        lines.append(f"[discovered.{name}]")
        lines.append(f"model_names = {toml_list(result.model_names)}")
        lines.append("")
    return "\n".join(lines)


def write_cache(results: dict[str, Discovered], path: Path = DISCOVERED_FILE) -> None:
    path = resolve(path)
    path.write_text(render_cache(results), encoding="utf-8")


def load_discovered(path: Path = DISCOVERED_FILE) -> dict[str, tuple[str, ...]]:
    path = resolve(path)
    raw = read_cache(path).get("discovered")
    if not isinstance(raw, dict):
        return {}
    out: dict[str, tuple[str, ...]] = {}
    for name, table in raw.items():
        if not isinstance(table, dict):
            continue
        names = table.get("model_names")
        if isinstance(names, list):
            out[name] = tuple(str(n) for n in names)
    return out


def find_extra_providers(
    registry: Registry,
    discovered: dict[str, tuple[str, ...]],
) -> dict[str, list[str]]:
    """Модели из models.toml, которые отдаёт ещё и другой провайдер.

    Только подсказка: дописывать второго провайдера молча нельзя, это тихо
    расширит маршрут и увеличит счёт.
    """
    extra: dict[str, list[str]] = {}
    for model_name, model in registry.models.items():
        for provider_name, names in discovered.items():
            if (
                provider_name in model.providers
                or provider_name not in registry.providers
            ):
                continue
            if model_name in names:
                extra.setdefault(model_name, []).append(provider_name)
    return extra


def merge_into_models(
    registry: Registry,
    discovered: dict[str, tuple[str, ...]],
    path: Path = MODELS_FILE,
    dry_run: bool = False,
) -> list[Model]:
    """Дописать в models.toml отсутствующие модели. Существующие не трогаем.

    Сравнение строго по имени модели. Новая модель получает того провайдера,
    у которого стоит раньше в providers.toml, — это единственный детерминированный
    выбор; остальные провайдеры с той же моделью попадают в find_extra_providers.
    """
    known = set(registry.models)
    added: list[Model] = []
    for provider_name in registry.providers:
        for model_name in discovered.get(provider_name, ()):
            if model_name in known:
                continue
            known.add(model_name)
            added.append(
                Model(
                    name=model_name,
                    providers=(provider_name,),
                    enabled=True,
                )
            )

    if not added or dry_run:
        return added

    path = resolve(path)
    body = [models_header(path), ""]
    for model in [*registry.models.values(), *added]:
        body.append("[[models]]")
        body.append(f"model_name = {toml_str(model.name)}")
        body.append(f"providers = {toml_list(model.providers)}")
        body.append(f"enabled = {'true' if model.enabled else 'false'}")
        body.append("")
    path.write_text("\n".join(body), encoding="utf-8")
    return added


def models_header(path: Path) -> str:
    """Шапка models.toml (всё до первого [[models]]), чтобы merge её не терял."""
    path = resolve(path)
    default = "# models/models.toml — модели, к которым обращается код."
    try:
        lines = path.read_text(encoding="utf-8").splitlines()
    except OSError:
        return default
    for index, line in enumerate(lines):
        if line.strip().startswith("[[models]]"):
            header = "\n".join(lines[:index]).rstrip()
            return header or default
    return default


def run_startup_discovery(
    registry: Registry | None = None,
    gateway: Gateway | None = None,
    path: Path = DISCOVERED_FILE,
    ttl: timedelta = TTL_CACHE_TTL,
    force: bool = False,
) -> dict[str, Discovered]:
    """Опрос на старте скрипта. Никогда не бросает исключений."""
    try:
        if registry is None:
            registry = load_registry()
        if not force and cache_is_fresh(path, ttl):
            log.info(
                "Кэш списка моделей свежий (age=%s), опрос пропущен", cache_age(path)
            )
            return {}
        owns_gateway = gateway is None
        gw = gateway if gateway is not None else Gateway(registry)
        try:
            results = poll_all(gw, registry)
        finally:
            if owns_gateway:
                gw.close()
        for name, result in sorted(results.items()):
            if result.ok:
                log.info(
                    "Провайдер %s: доступно моделей — %d", name, len(result.model_names)
                )
            else:
                log.warning("Провайдер %s: опрос не удался — %s", name, result.error)
        write_cache(results, path)
        return results
    except Exception as exc:  # noqa: BLE001 — старт не должен зависеть от сети
        log.warning("Опрос списка моделей пропущен: %s", exc)
        return {}


def print_report(registry: Registry, discovered: dict[str, tuple[str, ...]]) -> None:
    print("В models.toml:")
    if not registry.models:
        print("  (пусто)")
    for model in sorted(registry.models.values(), key=lambda m: m.name):
        flag = "" if model.enabled else "   [отключена]"
        print(f"  {model.label}{flag}   провайдеры: {', '.join(model.providers)}")

    print("\nНайдено у провайдеров:")
    if not discovered:
        print("  (пусто — запусти опрос: python model_discovery.py --poll)")
        return
    for provider_name in sorted(discovered):
        names = discovered[provider_name]
        fresh = [n for n in names if n not in registry.models]
        print(
            f"  {provider_name}: всего {len(names)}, не описано в models.toml {len(fresh)}"
        )
        for name in fresh:
            print(f"      + {name}")

    extra = find_extra_providers(registry, discovered)
    if extra:
        print("\nПодсказка: эти модели из models.toml отдаёт ещё и другой провайдер.")
        print(
            "Самостоятельно добавлять их не нужно — маршрут расширится и счёт вырастет."
        )
        for model_name, provider_names in sorted(extra.items()):
            print(f"  {model_name}: ещё есть у {', '.join(sorted(provider_names))}")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Опрос списка моделей у провайдеров и слияние в models.toml",
    )
    parser.add_argument(
        "--poll", action="store_true", help="принудительный опрос, игнорируя TTL кэша"
    )
    parser.add_argument(
        "--merge", action="store_true", help="дописать найденные модели в models.toml"
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="с --merge показать, что будет добавлено, не записывая файл",
    )
    parser.add_argument(
        "--log-level", default="INFO", help="уровень логирования (default: INFO)"
    )
    args = parser.parse_args(argv)

    logging.basicConfig(
        level=getattr(logging, args.log_level.upper(), logging.INFO),
        format="%(levelname)s %(name)s: %(message)s",
    )

    try:
        registry = load_registry()
    except ConfigError as exc:
        print(f"Ошибка конфигурации: {exc}", file=sys.stderr)
        return 2
    for warning in registry.warnings:
        print(f"ВНИМАНИЕ: {warning}", file=sys.stderr)

    discovered = load_discovered()
    if args.poll or not discovered:
        print("Опрос провайдеров...")
        run_startup_discovery(registry, force=args.poll)
        discovered = load_discovered()

    if args.merge:
        added = merge_into_models(registry, discovered, dry_run=args.dry_run)
        if not added:
            print("Добавлять нечего: все найденные модели уже есть в models.toml.")
        elif args.dry_run:
            print(f"Было бы добавлено моделей: {len(added)} (файл не тронут)")
            for model in added:
                print(f"  + {model.label}   провайдеры: {', '.join(model.providers)}")
        else:
            print(f"Добавлено моделей в models.toml: {len(added)}")
            for model in added:
                print(f"  + {model.label}   провайдеры: {', '.join(model.providers)}")
        print()
        print_report(registry, discovered)
        return 0

    print_report(registry, discovered)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
