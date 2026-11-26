# !/usr/bin/env python
# -*-coding:utf-8 -*-

"""
# File       : __init__.py
# Time       ：2023/7/28 17:02
# Author     ：yujia
# version    ：python 3.6
# Description：仓库根目录遗留的空包文件。

本仓库并不是一个可 `pip install` 的包：`src/`（推理层）与 `app/`（服务层）
各自是独立包，运行时把**项目根目录**放进 `sys.path` 即可，例如
`python service.py`、`pytest tests`、`uvicorn app.main:app` 都满足这个条件。

保留本文件仅为兼容历史上 `import Text_select_captcha_github` 的写法，不要在这里加逻辑。
"""
