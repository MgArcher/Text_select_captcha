# !/usr/bin/env python
# -*-coding:utf-8 -*-

"""日志：控制台 + 滚动文件（按大小切分，避免日志无限增长）。

`Text_select_captcha.*` 下所有 logger 都挂到这里，配置见 `configs/v8.toml` 的 `[log]`。
"""
from __future__ import annotations

import logging
import sys
from logging.handlers import RotatingFileHandler
from pathlib import Path
from typing import Any, Optional

from src.config import resolve

PKG = "Text_select_captcha"
FMT = "%(asctime)s | %(levelname)-7s | %(name)s | %(message)s"

_log_file: Optional[Path] = None


def setup_logging(cfg: dict[str, Any] | None = None, force: bool = False) -> Optional[Path]:
    """配置日志，返回日志文件路径；重复调用不会重复挂 handler。

    :param cfg:   `[log]` 段（dir / level / max_mb / backups）
    :param force: True 时清掉已有 handler 重新配置（测试用）
    """
    global _log_file
    cfg = cfg or {}
    logger = logging.getLogger(PKG)

    if logger.handlers and not force:
        return _log_file

    for handler in list(logger.handlers):
        logger.removeHandler(handler)
        handler.close()

    level = getattr(logging, str(cfg.get("level", "INFO")).upper(), logging.INFO)
    logger.setLevel(level)
    logger.propagate = False

    formatter = logging.Formatter(FMT)
    console = logging.StreamHandler(sys.stdout)
    console.setFormatter(formatter)
    logger.addHandler(console)

    _log_file = None
    try:
        log_dir = resolve(cfg.get("dir", "logs"))
        log_dir.mkdir(parents=True, exist_ok=True)
        path = log_dir / "app.log"
        file_handler = RotatingFileHandler(
            path,
            maxBytes=int(float(cfg.get("max_mb", 10)) * 1024 * 1024),
            backupCount=int(cfg.get("backups", 5)),
            encoding="utf-8",
        )
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)
        _log_file = path
    except OSError as exc:  # 目录不可写就只打控制台，不影响服务
        logger.warning("日志文件不可用，仅输出到控制台: %s", exc)

    logger.info("日志初始化 level=%s file=%s", logging.getLevelName(level), _log_file)
    return _log_file
