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
import traceback
from fastapi import APIRouter, HTTPException
from fastapi.responses import StreamingResponse

from app.models import Input, OutPut
from app.utils import operation
from app.utils import errors


router = APIRouter()

# @router.post("/", )
@router.post("/identify", response_model=OutPut, summary="识别结果")
async def identify_picture(Item: Input):
    """
   - **dataType: int；必须；文件类型：1:链接地址，2:文件字节流**
   - **imageSource: str；必须；源文件地址或源文件流，参照dataType，dataType为1需要传链接地址，dateType为2需要传文件流。传文件流方式，要base64编码，并去掉base64头标识。**
   - **imageID: str；不必须；图片名称或id**
    """
    try:
        res = await operation.run(Item)
    except:
        print("error：", traceback.format_exc())
        return errors.bad_error()
    return {'code': 200, 'msg': "成功", "data": res}


@router.post("/show", summary="识别后图片效果")
async  def show_result(Item: Input):
    """
   - **dataType: int；必须；文件类型：1:链接地址，2:文件字节流**
   - **imageSource: str；必须；源文件地址或源文件流，参照dataType，dataType为1需要传链接地址，dateType为2需要传文件流。传文件流方式，要base64编码，并去掉base64头标识。**
   - **imageID: str；不必须；图片名称或id**
    """
    try:
        img_bytes = await operation.run_show(Item)
    except:
        print("error：", traceback.format_exc())
        return errors.bad_error()
    return StreamingResponse(img_bytes, media_type="image/jpeg")


@router.get("/")
def ok():
    return {'code': 200, 'msg': "成功",  "data": "ok"}