"""Тесты llm-rotator.

Пакет, а не просто папка: тесты делят вспомогательные функции и должны
импортироваться как `tests.*`. Путь до корня репозитория добавляется в sys.path,
чтобы `import llm_rotator` работал и без установки пакета.

Запуск из корня репозитория:

    python -m unittest discover -s tests -t .
    python -m unittest tests.test_runner -v
"""

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
