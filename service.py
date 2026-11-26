# !/usr/bin/env python
# -*-coding:utf-8 -*-

"""
# File       : service.py
# Time       ：2021/9/26 14:36
# Author     ：yujia
# version    ：python 3.9+
# Description：服务启动入口（argparse + uvicorn）。

    python service.py                      # 默认读 configs/v8.toml 的 [server]
    python service.py --port 8002          # 换端口
    python service.py --eager              # 启动即加载模型
    uvicorn app.main:app                   # 等价写法（模块级 app 对象）
    uvicorn app.main:create_app --factory  # 工厂模式
"""
from __future__ import annotations

import argparse
from typing import Sequence

import uvicorn

from app import main as app_main
from src.config import load_config, section


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="service.py", description="文字点选验证码识别服务（v8）")
    p.add_argument("--config", default="v8", help="配置名或路径，默认 configs/v8.toml")
    p.add_argument("--host", default=None, help="监听地址，默认读配置 [server].host")
    p.add_argument("--port", type=int, default=None, help="监听端口，默认读配置 [server].port")
    p.add_argument("--eager", action="store_true", help="启动即加载模型（默认首次请求懒加载）")
    p.add_argument("--log-level", default="info", help="uvicorn 日志级别，默认 info")
    return p


def main(argv: Sequence[str] | None = None) -> None:
    args = build_parser().parse_args(argv)

    cfg = section(load_config(args.config), "server")
    host = args.host or str(cfg.get("host", "0.0.0.0"))
    port = args.port or int(cfg.get("port", 8000))

    # 绑定地址是 0.0.0.0（监听所有网卡），但打印给用户看的应该是能点的地址
    display_host = "localhost" if host in ("0.0.0.0", "::", "::0") else host

    print(f"📡 服务地址: http://{display_host}:{port}")
    print(f"📚 API文档: http://{display_host}:{port}/docs")
    print(f"📖 ReDoc文档: http://{display_host}:{port}/redoc")
    if display_host != host:
        print(f"   （实际监听 {host}:{port}，所有网卡）")

    application = app_main.create_app(eager=args.eager, config=args.config)
    uvicorn.run(application, host=host, port=port, log_level=args.log_level)


if __name__ == "__main__":
    main()
