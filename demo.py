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
s = time.time()
cap = TextSelectCaptcha()
print("加载模型耗时：", time.time() - s)

image_path = "docs/res.jpg"
s = time.time()

result = cap.run(image_path)
print(result)
print("耗时：", time.time() - s)
cap.yolo.infer(image_path)
print("生成图片res1.jpg")
drow_img(image_path, result)
print("生成图片res2.jpg")