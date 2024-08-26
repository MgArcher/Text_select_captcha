# !/usr/bin/env python
# -*-coding:utf-8 -*-

"""
# File       : captcha.py
# Time       ：2023/04/23 18:28
# Author     ：yujia
# version    ：python 3.6
# Description：
"""
from src.method.click_captcha_solver import ClickCaptchaSolver
from src.utils.utils import drow_img


class TextSelectCaptcha(object):
    def __init__(self, per_path='pre_model_v3.bin', yolo_path='best_v2.bin', sign=True):
        self.click_captcha_solver = ClickCaptchaSolver(per_path=per_path, yolo_path=yolo_path, sign=sign)
        self.yolo = self.click_captcha_solver.yolo

        self.method = {
            "click_captcha_solver": self.click_captcha_solver,
        }

    def run(self, image_path, method="click_captcha_solver"):
        result = self.method[method].run(image_path)
        return result


if __name__ == '__main__':
    cap = TextSelectCaptcha()
    image_path = r"../docs/res.jpg"
    result = cap.run(image_path)
    print(result)
    cap.yolo.infer(image_path)
    drow_img(image_path, result)