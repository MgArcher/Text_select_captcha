# !/usr/bin/env python
# -*-coding:utf-8 -*-

"""
# File       : jiyan.py
# Time       ：2024/8/17 12:43
# Author     ：yujia
# version    ：python 3.6
# Description：
文字点选
"""
from fastapi.responses import Response
from fastapi import APIRouter, Depends
from app.models.input import Input

from app.services import operation

router = APIRouter()

@router.post("/identify", summary="图文点选识别")
async def identify(Item: Input):
    result = await operation.run(Item)
    return {"code": 200, "msg": "成功", "data": result}


@router.post("/show", summary="识别后图片效果")
async def show_result(Item: Input):
    img_bytes = await operation.run_show(Item)
    return Response(content=img_bytes, media_type="image/jpeg")