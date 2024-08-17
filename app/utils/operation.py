# !/usr/bin/env python
# -*-coding:utf-8 -*-

"""
# File       : model.py
# Time       ：2021/9/26 15:26
# Author     ：yujia
# version    ：python 3.6
# Description：
"""
"""此处实例化模型类"""
from src import captcha
from app.utils import interface

cap_model = captcha.TextSelectCaptcha()


def run(item):
    imageID = item.imageID
    imageSource = interface.set_imageSource(item.dataType, item.imageSource)
    res = cap_model.run(imageSource)
    return {"imageID": imageID, "res": res}


def run_show(item):
    imageSource = interface.set_imageSource(item.dataType, item.imageSource)
    results = cap_model.run(imageSource)
    img_bytes = interface.drow_img(imageSource, results)
    return img_bytes