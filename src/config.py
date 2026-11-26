# !/usr/bin/env python
# -*-coding:utf-8 -*-

"""
# File       : config.py
# Time       ：2026/11/3 11:20
# Description：v8 配置加载（configs/v8.toml），路径统一相对项目根目录解析。
"""
from __future__ import annotations

import os
from pathlib import Path
from typing import Any

try:  # Python 3.11+
    import tomllib
except ImportError:  # pragma: no cover - Python 3.8~3.10 需要 pip install tomli
    import tomli as tomllib

# <project>/src/config.py -> <project>
ROOT = Path(__file__).resolve().parents[1]
CONFIG_DIR = ROOT / "configs"

MAX_AUTO_THREADS = 8  # 上限：再多线程收益明显递减，反而抢核


def resolve_threads(value: Any = None) -> int:
    """ONNX Runtime 线程数。

    显式正整数 -> 用它；`"auto"` / 0 / 缺省 -> 按 CPU 核数取（上限 `MAX_AUTO_THREADS`）。
    核数少的小机器按核数给，避免线程数超过物理核反而变慢。
    """
    if isinstance(value, bool):  # bool 是 int 的子类，先排除
        raise ValueError(f"threads 不能是布尔值: {value}")
    if isinstance(value, int) and value > 0:
        return value
    if isinstance(value, str) and value.strip() and value.strip().lower() != "auto":
        explicit = int(value)
        if explicit > 0:
            return explicit
    return max(1, min(MAX_AUTO_THREADS, os.cpu_count() or 1))


def resolve(rel: str | Path) -> Path:
    """把相对路径解析到项目根目录下；绝对路径原样返回。"""
    p = Path(rel)
    return p if p.is_absolute() else (ROOT / p)


def load_config(name_or_path: str | Path = "v8") -> dict[str, Any]:
    """按名称（configs/<name>.toml）或显式路径加载 TOML 配置。"""
    p = Path(name_or_path)
    if not p.suffix:
        p = CONFIG_DIR / f"{p}.toml"
    p = resolve(p)
    if not p.exists():
        raise FileNotFoundError(f"配置文件不存在: {p}")
    return tomllib.loads(p.read_text(encoding="utf-8"))


def section(cfg: dict[str, Any], key: str) -> dict[str, Any]:
    """读取一个 [section]，缺省返回空字典。"""
    value = cfg.get(key, {})
    if not isinstance(value, dict):
        raise ValueError(f"配置段 [{key}] 必须是 table")
    return value
