# !/usr/bin/env python
# -*-coding:utf-8 -*-

"""Web 服务层：HTTP 协议、路由、模型、业务编排。

模型的加载与计算全在 `src/`（推理层）；依赖单向 `app -> src`，
`src/` 不允许 import `app/`，也不允许 import FastAPI。
"""
__version__ = "2.1.0"
