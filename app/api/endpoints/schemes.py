# !/usr/bin/env python
# -*-coding:utf-8 -*-

"""
# File       : schemes.py
# Time       ：2026/11/4 10:30
# Author     ：yujia
# version    ：python 3.9+
# Description：方案列表。
"""
from __future__ import annotations

from fastapi import APIRouter

from app.services import operation
from app.utils.errors import ok

router = APIRouter()


@router.get("/schemes", summary="已加载方案")
async def list_schemes():
    return ok(operation.schemes())
