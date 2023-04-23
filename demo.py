# !/usr/bin/env python
# -*-coding:utf-8 -*-

"""
# File       : demo.py
# Time       ：2023/4/21 14:56
# Author     ：yujia
# version    ：python 3.6
# Description：
"""
from src.captcha import TextSelectCaptcha, drow_img
import time

cap = TextSelectCaptcha()

image_path = "errot/1682234534.jpg"

s = time.time()
result = cap.run(image_path)
print(result)
print("耗时：", time.time() - s)
cap.yolo.infer(image_path)
drow_img(image_path, result)
