# -*- coding: utf-8 -*-
"""
check_imports.py
AST-линтер архитектурных правил импортов.

Правила слоёв (повышающаяся зависимость сверху вниз):
  core_*         ← можно импортировать всем
  impl_*         ← можно импортировать service_*, scripts_*, но НЕ strategy_*
  service_*      ← можно импортировать scripts_*, но НЕ impl_* сверху вниз (обратных зависимостей быть не должно)
  strategy_*     ← может импортировать только core_* (и внешние пакеты)
  scripts_*      ← точки входа: могут импортировать service_* и core_*, но НЕ impl_* напрямую и НЕ strategy_* (они получают стратегию через DI)

Так как репозиторий плоский, классификация по префиксам имён файлов. Если позже появятся директории,
скрипт легко адаптируется.
"""

from __future__ import annotations

import os
import ast
import sys
from typing import Dict, List, Tuple, Set, Optional


LAYER_BY_PREFIX = {
    "core_": "core",
    "impl_": "impl",
    "service_": "service",
    "strategy_": "strategy",
    "script_": "scripts",
}

# Разрешённые импорты по слоям
ALLOWED: Dict[str, Set[str]] = {
    "core": {"core"},
    "impl": {"core", "impl"},
    "service": {"core", "impl", "service"},
    "strategy": {"core"},
    "scripts": {"core", "service"},
}

# Модули, которые считаем внешними (разрешены везде)
EXTERNAL_WHITELIST_PREFIXES = (
    "numpy", "pandas", "scipy", "sklearn", "matplotlib", "ta", "decimal",
    "pydantic", "typing", "dataclasses", "enum", "collections", "time", "datetime",
    "pathlib", "functools", "itertools", "json", "yaml", "asyncio", "logging", "os", "sys",
    "re", "importlib", "inspect", "argparse", "typing_extensions",
)

def detect_layer(filename: str) -> Optional[str]:
    rel = filename.lstrip("./")
    if rel.startswith("strategies/"):
        return "strategy"
    base = os.path.basename(rel)
    for pref, layer in LAYER_BY_PREFIX.items():
        if base.startswith(pref):
            return layer
    if base in ("prepare_and_run.py", "app.py"):
        return "scripts"
    # impl: известные имена существующих реализаций
    if base in ("execution_sim.py", "binance_ws.py", "binance_public.py", "sim_adapter.py",
                "quantizer.py", "fees.py", "slippage.py", "latency.py", "risk.py", "risk_guard.py", "event_bus.py"):
        return "impl"
    # core: новые core_* + существующие core_constants.*
    if base.startswith("core") or base in ("core_constants.py", "coreprice_scale.py"):
        return "core"
    return None


def iter_py_files(root: str) -> List[str]:
    result = []
    for dirpath, dirnames, filenames in os.walk(root):
        dirnames[:] = [d for d in dirnames if d != "__pycache__"]
        for f in filenames:
            if f.endswith(".py"):
                result.append(os.path.join(dirpath, f))
    return result


def analyze_file(path: str) -> List[Tuple[str, str]]:
    src = open(path, "r", encoding="utf-8").read()
    try:
        tree = ast.parse(src)
    except SyntaxError:
        # пропускаем файлы с синтаксическими ошибками
        return []
    src_layer = detect_layer(path)
    if not src_layer:
        return []

    violations: List[Tuple[str, str]] = []

    def target_layer_of(module_name: str) -> Optional[str]:
        base = module_name.split(".")[0] + ".py"
        return detect_layer(base)

    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                mod = alias.name
                # внешние пакеты — пропускаем
                if mod.startswith(EXTERNAL_WHITELIST_PREFIXES):
                    continue
                tgt = target_layer_of(mod)
                if tgt and tgt not in ALLOWED[src_layer]:
                    violations.append((mod, f"{src_layer} → {tgt} запрещено"))
        elif isinstance(node, ast.ImportFrom):
            if node.module is None:
                continue
            mod = node.module
            if mod.startswith(EXTERNAL_WHITELIST_PREFIXES):
                continue
            tgt = target_layer_of(mod)
            if tgt and tgt not in ALLOWED[src_layer]:
                violations.append((mod, f"{src_layer} → {tgt} запрещено"))
    return violations


def main() -> int:
    root = "."
    files = iter_py_files(root)
    total = 0
    had = 0
    for f in files:
        total += 1
        vios = analyze_file(f)
        if vios:
            had += 1
            print(f"[FAIL] {os.path.basename(f)}:")
            for mod, reason in vios:
                print(f"    import {mod!s}  →  {reason}")
    if had == 0:
        print("OK: нарушений архитектурных импортов не найдено")
        return 0
    else:
        print(f"Найдено файлов с нарушениями: {had}/{total}")
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
