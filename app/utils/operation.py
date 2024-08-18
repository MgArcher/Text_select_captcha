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
    centre = lambda x1, y1, x2, y2: [(x1 + x2) / 2, (y1 + y2) / 2]
    res_centre = [centre(*i) for i in res]
    return {"imageID": imageID, "res": {"crop": res, "crop_centre": res_centre}}


def run_show(item):
    imageSource = interface.set_imageSource(item.dataType, item.imageSource)
    results = cap_model.run(imageSource)
    img_bytes = interface.drow_img(imageSource, results)
    return img_bytes