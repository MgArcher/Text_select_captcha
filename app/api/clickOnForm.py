# !/usr/bin/env python
# -*-coding:utf-8 -*-

"""
# File       : clickOnData.py
# Time       ：2021/9/26 17:08
# Author     ：yujia
# version    ：python 3.6
# Description：
"""
# !/usr/bin/env python
# -*-coding:utf-8 -*-

"""
# File       : clickOn.py
# Time       ：2021/9/26 14:31
# Author     ：yujia
# version    ：python 3.6
# Description：
"""
from fastapi import Form
from fastapi_restful import Resource

from app.utils.model import cap_model
from app.utils import interface


class ClickOnForm(Resource):

    def post(self,
               dataType: int = Form(..., example=1),
               imageSource: str = Form(..., example="https://static.geetest.com/captcha_v3/batch/v3/2021-09-18T10/word/182ebfd7468d42f48aed95a83a4fcef5.jpg?challenge=5e8cc0a940dbf1fa0c5fb2ce0a0172eb"),
               imageID: str = Form(None, example="asdr-jxckne"),
               ):
        """

dataType: dataType为1时imageSource接收链接地址，为2时imageSource接收base64编码后的文件流
imageSource：base64编码后的文件流，或链接地址
        "dataType": 1
        "imageSource": ,
        "imageID": "string"
        """
        results = dict()
        data = {
            "dataType": dataType,
            "imageSource": imageSource,
            "imageID": imageID
        }
        imageSource = interface.set_imageSource(data)
        res = cap_model.run(imageSource)
        results['data'] = res
        results['imageID'] = data.get("imageID", "")
        return results