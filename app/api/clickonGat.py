# !/usr/bin/env python
# -*-coding:utf-8 -*-

"""
# File       : clickGet.py
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
from fastapi_restful import Resource

from app.utils.model import cap_model
from app.utils import interface


class ClickOnGat(Resource):

    def get(self, imageSource: str, imageID: str = ""):
        """
接收json请求，识别点选验证码
imageSource接收链接地址
        """
        results = dict()
        data = {
            "imageSource": imageSource,
            "dataType": 1,
            "imageID": imageID
        }
        imageSource = interface.set_imageSource(data)
        res = cap_model.run(imageSource)
        results['data'] = res
        results['imageID'] = data.get("imageID", "")
        return results