"""Разрешаем тестам импортировать учебный пакет из корня проекта."""

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
