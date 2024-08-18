# !/usr/bin/env python
# -*-coding:utf-8 -*-

"""
# File       : interface.py
# Time       ：2021/5/6 15:44
# Author     ：yujia
# version    ：python 3.6
# Description：
"""
import io
import random
import base64

import aiohttp
import cv2
import numpy as np
from PIL import Image


async def fetch(url):
    timeout = aiohttp.ClientTimeout(
        total=10,  # 请求的总超时（秒）
        connect=5,  # 连接超时（秒）
        sock_read=5,  # 套接字读取超时（秒）
        sock_connect=5  # 套接字连接超时（秒）
    )
    async with aiohttp.ClientSession() as session:
        async with session.get(url, ssl=False, timeout=timeout) as response:
            return await response.read()

# 图片读取操作
async def set_imageSource(dataType, imageSource):
    if dataType == 1:
        imageSource = await fetch(imageSource)
    else:
        imageSource = base64.b64decode(bytes(imageSource, 'utf-8'))
    return imageSource


# 图片绘制
def drow_img(image_path,result):
    img = Image.open(io.BytesIO(image_path))
    img = img.convert("RGB")
    img = np.array(img)

    def plot_one_box(x, img, color=None, label=None, line_thickness=None):
        # Plots one bounding box on image img
        tl = line_thickness or round(0.002 * (img.shape[0] + img.shape[1]) / 2) + 1  # line/font thickness
        color = color or [random.randint(0, 255) for _ in range(3)]
        c1, c2 = (int(x[0]), int(x[1])), (int(x[2]), int(x[3]))
        cv2.rectangle(img, c1, c2, color, thickness=tl, lineType=cv2.LINE_AA)
        if label:
            tf = max(tl - 1, 1)  # font thickness
            t_size = cv2.getTextSize(label, 0, fontScale=tl / 3, thickness=tf)[0]
            c2 = c1[0] + t_size[0], c1[1] - t_size[1] - 3
            cv2.rectangle(img, c1, c2, color, -1, cv2.LINE_AA)  # filled
            cv2.putText(img, label, (c1[0], c1[1] - 2), 0, tl / 3, [225, 255, 255], thickness=tf, lineType=cv2.LINE_AA)
        return img

    colors = [[random.randint(0, 255) for _ in range(3)] for _ in range(len(result))]
    for i, xyxy in enumerate(result):
        label = i + 1
        img = plot_one_box(xyxy, img, label=str(label), color=colors[i], line_thickness=1)
    large_img = Image.fromarray(img)
    img_bytes = io.BytesIO()
    large_img.save(img_bytes, format='jpeg')
    img_bytes.seek(0)
    return img_bytes
