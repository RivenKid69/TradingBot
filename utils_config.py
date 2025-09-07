# -*- coding: utf-8 -*-
"""
services/utils_config.py
Утилиты для работы с конфигами: сохранение снапшота запуска.
"""

from __future__ import annotations

import os
import shutil
import time
from typing import Optional

def snapshot_config(config_path: str, artifacts_dir: str) -> Optional[str]:
    """
    Копирует файл конфига в папку артефактов под именем artifact_config_<ts>.yaml.
    Возвращает путь к копии или None при ошибке.
    """
    try:
        os.makedirs(artifacts_dir, exist_ok=True)
        ts = int(time.time())
        dst = os.path.join(artifacts_dir, f"artifact_config_{ts}.yaml")
        shutil.copyfile(config_path, dst)
        return dst
    except Exception:
        return None
