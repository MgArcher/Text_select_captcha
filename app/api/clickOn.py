# !/usr/bin/env python
# -*-coding:utf-8 -*-

"""
# File       : clickOn.py
# Time       ：2021/9/26 14:31
# Author     ：yujia
# version    ：python 3.6
# Description：
"""
from fastapi_restful import Resource, set_responses
from pydantic import BaseModel
from typing import Optional

from app.utils.model import cap_model
from app.utils import interface


class Input(BaseModel):
    dataType: int
    imageSource: str
    imageID: Optional[str] = None

    class Config:
        schema_extra = {
            "example": {
                "dataType": 1,
                "imageSource": "https://static.geetest.com/captcha_v3/batch/v3/2021-09-18T10/word/182ebfd7468d42f48aed95a83a4fcef5.jpg?challenge=5e8cc0a940dbf1fa0c5fb2ce0a0172eb",
                "imageID": "string"
                }
        }


class ClickOn(Resource):


    def post(self, item: Input):
        """
接收json请求，识别点选验证码
dataType: dataType为1时imageSource接收链接地址，为2时imageSource接收base64编码后的文件流
imageSource：base64编码后的文件流，或链接地址
        """
        results = dict()
        data = item.dict()
        imageSource = interface.set_imageSource(data)
        res = cap_model.run(imageSource)
        results['data'] = res
        results['imageID'] = data.get("imageID", "")
        return results