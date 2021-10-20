# !/usr/bin/env python
# -*-coding:utf-8 -*-

"""
# File       : clickOnFile.py
# Time       ：2021/9/26 15:14
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
from fastapi_restful import Resource
from fastapi import File, UploadFile

from app.utils.model import cap_model


class ClickOnFile(Resource):

    def post(self,
             file: UploadFile = File(..., example="上传的图片")):
        """
        接收图片文件，识别点选验证码
        """
        results = dict()
        content_type = file.content_type
        results['imageID'] = file.filename
        if "image" in content_type:
            contents = file.file.read()
            res = cap_model.run(contents)
            results['data'] = res
        else:
            results = {"error": "上传的不是图片！"}
        return results