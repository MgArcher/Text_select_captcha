# !/usr/bin/env python
# -*-coding:utf-8 -*-

"""
# File       : clickOn.py
# Time       ：2021/9/26 14:31
# Author     ：yujia
# version    ：python 3.6
# Description：
"""
import traceback
from fastapi_restful import Resource, set_responses
from pydantic import BaseModel
from typing import Optional, Union

from app.utils.model import cap_model
from app.utils import interface
from app.utils import errors


class Input(BaseModel):
    dataType: int
    imageSource: str
    imageID: Optional[str] = None

    class Config:
        schema_extra = {
            "example": {
                "dataType": 1,
                "imageSource": "https://static.geetest.com/captcha_v3/batch/v3/33114/2023-04-23T10/word/2c4d5ce6e02e4cb883f45e6476be9ba7.jpg?challenge=b11d16e456aad5c622d244047cc389ea",
                "imageID": "string"
                }
        }


class OutPut(BaseModel):
    code: str
    msg: str
    data: Union[list, dict, str]

    class Config:
        schema_extra = {
            "example": {
                      "code": "200",
                      "msg": "成功",
                      "data": {
                        "imageID": "string",
                        "res": [
                          [
                            184,
                            0,
                            259,
                            67
                          ],
                          [
                            176,
                            238,
                            244,
                            310
                          ],
                          [
                            63,
                            70,
                            132,
                            142
                          ]
                        ]
                      }
                    }
        }




class ClickOn(Resource):
    output_model = OutPut

    def run(self, data):
        imageID = data.get("imageID")
        imageSource = interface.set_imageSource(data)
        res = cap_model.run(imageSource)
        return {"imageID": imageID, "res": res}

    async def post(self, item: Input):
        """
        - **dataType: int；必须；文件类型：1:链接地址，2:文件字节流**
        - **imageSource: str；必须；源文件地址或源文件流，参照dataType，dataType为1需要传链接地址，dateType为2需要传文件流。传文件流方式，要base64编码，并去掉base64头标识。**
        - **imageID: str；不必须；图片名称或id**
        """
        data = item.dict()
        try:
            res = self.run(data)
        except:
            print("error：", traceback.format_exc())
            return errors.bad_error()
        return {'code': 200, 'msg': "成功",  "data": res}