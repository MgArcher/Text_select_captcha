# !/usr/bin/env python
# -*-coding:utf-8 -*-

"""
# File       : captcha.py
# Time       ：2023/04/23 18:28
# Author     ：yujia
# version    ：python 3.6
# Description：
"""
from src.method import jy_click
from src.utils.utils import drow_img


class TextSelectCaptcha(object):
    def __init__(self):
        self.jy_click = jy_click.JYClick()
        self.yolo = self.jy_click.yolo

        self.method = {
            "jy_click": self.jy_click,
        }

    def run(self, image_path, method="jy_click"):
        result = self.method[method].run(image_path)
        return result


if __name__ == '__main__':
    cap = TextSelectCaptcha()
    image_path = r"../docs/res.jpg"
    result = cap.run(image_path)
    print(result)
    cap.yolo.infer(image_path)
    drow_img(image_path, result)