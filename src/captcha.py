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
import numpy as np
from PIL import Image, ImageDraw, ImageFont

from src.utils import ver_onnx
from src.utils import yolo_onnx

from src.utils.ver_onnx import drow_img

save_path = os.path.join(os.path.dirname(__file__), '../model')
path = lambda a, b: os.path.join(a, b)

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


def make_char(text):

    # 创建图像
    image_width = 32
    image_height = 32
    background_color = (255, 255, 255)  # 白色背景
    text_color = (0, 0, 0)  # 黑色文本
    font_size = 32
    # text = "你"

    image = Image.new("RGB", (image_width, image_height), background_color)
    draw = ImageDraw.Draw(image)

    # 加载字体
    font_path = path(save_path, "simsun.ttc")# 字体文件路径
    font = ImageFont.truetype(font_path, font_size)

    # 计算文本位置居中
    text_width, text_height = draw.textsize(text, font=font)
    x = (image_width - text_width) // 2
    y = (image_height - text_height) // 2

    # 绘制文本
    draw.text((x, y), text, font=font, fill=text_color)
    # # 保存图像
    # image.save("output.png")
    return image


class TextSelectCaptcha(object):
    def __init__(self, per_path='pre_model_v2.bin', yolo_path='best_v2.bin', sign=True):
        """
        :param per_path: 识别模型文件路径
        :param yolo_path: 检测模型文件路径
        :param sign: 传入状态判断
        """
        per_path = path(save_path, per_path)
        yolo_path = path(save_path, yolo_path)
        if sign:
            from src.utils.load import decryption
            yolo_path = decryption(yolo_path)
            per_path = decryption(per_path)
        self.yolo = yolo_onnx.YOLOV5_ONNX(yolo_path, classes=['target', 'title', 'char'], providers=['CPUExecutionProvider'])
        self.pre = ver_onnx.PreONNX(per_path, providers=['CPUExecutionProvider'])

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
            chars = []
            for char in input_chars:
                # 判断是否为路径来决定是打开图片还是生成图片
                if os.path.isabs(char):
                    chars.append(open_image(char))
                else:
                    chars.append(make_char(char))
        else:
            chars = [i.get("crop") for i in data if i.get("classes") == "char"]
            # 根据坐标进行排序
            chars.sort(key=lambda x: x[0])
            chars = [img.crop(char) for char in chars]

        result = []
        for m, img_char in enumerate(chars):
            slys = []
            if len(targets) == 0:
                break
            elif len(targets) == 1:
                slys_index = 0
            else:
                for n, target in enumerate(targets):
                    img_target = img.crop(target)
                    similarity = self.pre.reason(img_char, img_target)
                    slys.append(similarity)
                print(slys)
                slys_index = slys.index(max(slys))
            result.append(targets[slys_index])
            targets.pop(slys_index)
            if len(targets) == 0:
                break
        return result

    def fex_run(self, image_path):
        img = open_image(image_path)
        data = self.yolo.decect(image_path)
        res = [
            (0, 0, 55, 40), (55, 0, 55, 40), (110, 0, 55, 40),
            (0, 40, 55, 40), (55, 40, 55, 40),(110, 40, 55, 40),
            (0, 80, 55, 40), (55, 80, 55, 40), (110, 80, 55, 40),
        ]
        data.extend(
            [{'crop': self.remove_whitespace(img, i), 'classes': 'target', 'prob': 0.8} for i in res]
        )
        # 需要选择的字
        targets = [i.get("crop") for i in data if i.get("classes") == "target"]
        targets = list(filter(lambda x: x is not None, targets))
        targets_copy = targets.copy()
        chars = [i.get("crop") for i in data if i.get("classes") == "char"]
        # 根据坐标进行排序
        chars.sort(key=lambda x: x[0])
        chars = [img.crop(char) for char in chars]

        result = []
        if targets:
            len_t = len(chars) - 1
            for i, img_char in enumerate(chars):
                slys = []
                if len(targets) == 1:
                    slys_index = 0
                else:
                    if -(len_t - i) != 0:
                        targets_ = targets[:-(len_t - i)]
                    else:
                        targets_ = targets
                    for target in targets_:
                        img_target = img.crop(target)
                        similarity = self.pre.reason(img_char, img_target)
                        slys.append(similarity)
                    slys_index = slys.index(max(slys))
                result.append(targets[slys_index])
                targets = targets[slys_index + 1:]

        kk = []
        for r in result:
            if r in targets_copy:
                kk.append(str(targets_copy.index(r)))
        order = "".join(kk)
        return result, order


if __name__ == '__main__':
    cap = TextSelectCaptcha()
    image_path = r"../docs/res.jpg"
    result = cap.run(image_path)
    print(result)
    cap.yolo.infer(image_path)
    drow_img(image_path, result)