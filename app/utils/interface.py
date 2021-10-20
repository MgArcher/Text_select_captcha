# !/usr/bin/env python
# -*-coding:utf-8 -*-

"""
# File       : interface.py
# Time       ：2021/5/6 15:44
# Author     ：yujia
# version    ：python 3.6
# Description：
"""
import requests
import base64


# 图片读取操作
def set_imageSource(data):
    if 'dataType' in data.keys() and data['dataType'] == 1:
        imageSource = requests.get(data['imageSource']).content
    else:
        imageSource = base64.b64decode(bytes(data['imageSource'], 'utf-8'))
    return imageSource

