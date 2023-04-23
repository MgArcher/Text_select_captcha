# !/usr/bin/env python
# -*-coding:utf-8 -*-

"""
# File       : clickOn.py
# Time       ：2021/9/26 14:31
# Author     ：yujia
# version    ：python 3.6
# Description：
"""
import io
import random

import cv2
import numpy as np
from PIL import Image
from fastapi_restful import Resource, set_responses
from pydantic import BaseModel
from typing import Optional
from fastapi.responses import StreamingResponse

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
                "imageSource": "https://static.geetest.com/captcha_v3/batch/v3/33114/2023-04-23T10/word/2c4d5ce6e02e4cb883f45e6476be9ba7.jpg?challenge=b11d16e456aad5c622d244047cc389ea",
                "imageID": "string"
                }
        }



def drow_img(image_path,result):
    img = Image.open(io.BytesIO(image_path))
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

class ClickOnFile(Resource):


    def post(self, item: Input):
        """
        - **dataType: int；必须；文件类型：1:链接地址，2:文件字节流**
        - **imageSource: str；必须；源文件地址或源文件流，参照dataType，dataType为1需要传链接地址，dateType为2需要传文件流。传文件流方式，要base64编码，并去掉base64头标识。**
        - **imageID: str；不必须；图片名称或id**
        """
        data = item.dict()
        imageSource = interface.set_imageSource(data)
        results = cap_model.run(imageSource)
        img_bytes = drow_img(imageSource, results)

        return StreamingResponse(img_bytes, media_type="image/jpeg")