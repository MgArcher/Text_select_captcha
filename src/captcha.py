# !/usr/bin/env python
# -*-coding:utf-8 -*-

"""
# File       : captcha.py
# Time       ：2023/04/23 18:28
# Author     ：yujia
# version    ：python 3.6
# Description：
"""

import os
from io import BytesIO
from PIL import Image
import numpy as np

from src.utils import ver_onnx
from src.utils import yolo_onnx
from src.utils.load import decryption


def open_image(file):
    if isinstance(file, np.ndarray):
        img = Image.fromarray(file)
    elif isinstance(file, bytes):
        img = Image.open(BytesIO(file))
    elif isinstance(file, Image.Image):
        img = file
    else:
        img = Image.open(file)
    img = img.convert('RGB')
    return img


class TextSelectCaptcha(object):
    def __init__(self, per_path='pre_model.bin', yolo_path='best.bin'):
        save_path = os.path.join(os.path.dirname(__file__), '../model')

        path = lambda a, b: os.path.join(a, b)
        per_path = path(save_path, per_path)
        yolo_path = path(save_path, yolo_path)

        self.yolo = yolo_onnx.YOLOV5_ONNX(decryption(yolo_path), classes=['target', 'title', 'char'], providers=['CPUExecutionProvider'])
        self.pre = ver_onnx.PreONNX(decryption(per_path), providers=['CPUExecutionProvider'])

    def run(self, image_path, input_chars=None):
        """
        检测
        :param img: 图片的路径、二进制数据或图片矩阵
        :return: list ---> [{'crop': [x1, y1, x2, y2], 'classes': ''}
        """
        img = open_image(image_path)
        data = self.yolo.decect(image_path)
        # 需要选择的字
        targets = [i.get("crop") for i in data if i.get("classes") == "target"]
        # 下方的字
        if input_chars:
            chars = input_chars
            chars = [open_image(char) for char in chars]
        else:
            chars = [i.get("crop") for i in data if i.get("classes") == "char"]
            # 根据坐标进行排序
            chars.sort(key=lambda x: x[0])
            chars = [img.crop(char) for char in chars]

        result = []
        for img_char in chars:
            slys = []
            if len(targets) == 1:
                slys_index = 0
            else:
                for target in targets:
                    img_target = img.crop(target)
                    similarity = self.pre.reason(img_char, img_target)
                    slys.append(similarity)
                slys_index = slys.index(max(slys))
            result.append(targets[slys_index])
            targets.pop(slys_index)
        return result


if __name__ == '__main__':
    cap = TextSelectCaptcha()
    image_path = r"../docs/res.jpg"
    result = cap.run(image_path)
    print(result)
    cap.yolo.infer(image_path)
    ver_onnx.drow_img(image_path, result)