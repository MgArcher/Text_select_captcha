# !/usr/bin/env python
# -*-coding:utf-8 -*-

"""
# File       : router.py
# Time       ：2024/8/14 16:55
# Author     ：yujia
# version    ：python 3.9+
# Description：v1 路由聚合（由 `app.main` 统一加 /api/v1 前缀）。
"""
from fastapi import APIRouter

from app.api.endpoints import dianxuan, schemes

router = APIRouter()
router.include_router(dianxuan.router, tags=["识别"])
router.include_router(schemes.router, tags=["方案"])
