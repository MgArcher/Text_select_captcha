# !/usr/bin/env python
# -*-coding:utf-8 -*-

"""
# File       : demo.py
# Time       ：2023/4/21 14:56
# Author     ：yujia
# version    ：python 3.6
# Description：
"""
from src.captcha import TextSelectCaptcha
from src.drawing import drow_img
import time
import json


s = time.time()
cap = TextSelectCaptcha()
print("加载模型耗时：", time.time() - s)

image_path = "docs/res.jpg"
image_path = r"D:\captcha\c972363d448b4400a42be924bb7577f2.jpg"
s = time.time()

result = cap.run(image_path)
print(f"推理耗时：{int((time.time() - s) *1000)}ms", )
print("文字坐标：", result, f"耗时：{int((time.time() - s) * 1000)}ms", )
drow_img(image_path, result)
print("生成图片res2.jpg")
result = cap.run_dict(image_path)
data = json.dumps(result, indent=4, ensure_ascii=False)
print(data)