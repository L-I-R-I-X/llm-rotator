"""Клиенты сервиса (`models/clients.toml`): токен → пространство имён.

У каждого харнесса свой токен, а значит своё `namespace` и свои удалённые чаты:
два харнесса с одинаковыми сообщениями никогда не делят чат.
"""

from __future__ import annotations

import hmac
import tomllib
from dataclasses import dataclass
from pathlib import Path

from ..registry import ConfigError, Registry

CLIENTS_FILE = Path("models") / "clients.toml"
CHAIN_PREFIX = "chain:"
DEFAULT_CLIENT = "default"


@dataclass(frozen=True, slots=True)
class Client:
    name: str
    default_model: str = ""
    """Модель или `chain:<имя>` на случай, если клиент модель не указал/ошибся."""


class ClientTable:
    def __init__(
        self,
        by_token: dict[str, Client] | None = None,
        open_client: Client | None = None,
    ) -> None:
        self._by_token = by_token or {}
        self.open_client = open_client
        """Единственный клиент без токена: когда clients.toml нет."""

    @property
    def names(self) -> tuple[str, ...]:
        if self.open_client:
            return (self.open_client.name,)
        return tuple(c.name for c in self._by_token.values())

    @property
    def requires_token(self) -> bool:
        return self.open_client is None

    def authenticate(self, token: str) -> Client | None:
        """Клиент по токену. Сравнение за постоянное время."""
        found: Client | None = None
        for known, client in self._by_token.items():
            if hmac.compare_digest(known.encode(), token.encode()):
                found = client
        return found


def check_model_ref(registry: Registry, ref: str, where: str) -> None:
    if ref.startswith(CHAIN_PREFIX):
        registry.chain(ref.removeprefix(CHAIN_PREFIX))
    elif ref not in registry.models:
        raise ConfigError(f"{where}: модель {ref!r} не найдена в models.toml")


def load_clients(
    base_dir: Path, env: dict[str, str], registry: Registry
) -> ClientTable:
    path = base_dir / CLIENTS_FILE
    if not path.exists():
        return ClientTable(open_client=Client(DEFAULT_CLIENT))
    try:
        doc = tomllib.loads(path.read_text(encoding="utf-8"))
    except (OSError, tomllib.TOMLDecodeError) as exc:
        raise ConfigError(f"{path.name}: {exc}") from None
    raw = doc.get("clients")
    if not isinstance(raw, list) or not raw:
        raise ConfigError(f"{path.name}: нужен непустой список [[clients]]")
    by_token: dict[str, Client] = {}
    names: set[str] = set()
    for index, table in enumerate(raw, start=1):
        where = f"{path.name} [[clients]] #{index}"
        if not isinstance(table, dict):
            raise ConfigError(f"{where}: ожидается таблица")
        name = table.get("client_name")
        token_env = table.get("token_env")
        if not isinstance(name, str) or not name.strip():
            raise ConfigError(f"{where}: нужно поле 'client_name'")
        if not isinstance(token_env, str) or not token_env.strip():
            raise ConfigError(f"{where}: нужно поле 'token_env'")
        name = name.strip()
        if name in names:
            raise ConfigError(f"{where}: клиент {name!r} описан дважды")
        token = env.get(token_env.strip(), "")
        if not token:
            raise ConfigError(f"{where}: в .env нет переменной {token_env}")
        if token in by_token:
            raise ConfigError(
                f"{where}: токен {token_env} совпал с токеном другого клиента"
            )
        default = str(table.get("default_model") or "").strip()
        if default:
            check_model_ref(registry, default, where)
        names.add(name)
        by_token[token] = Client(name, default)
    return ClientTable(by_token)
