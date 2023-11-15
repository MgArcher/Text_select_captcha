# !/usr/bin/env python
# -*-coding:utf-8 -*-

"""
# File       : utils.py
# Time       ：2023/11/13 17:02
# Author     ：yujia
# version    ：python 3.6
# Description：
"""
import os
from io import BytesIO
import random

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont

# 加载字体
try:
    font_path = "simsun.ttc"# 字体文件路径
    font = ImageFont.truetype(font_path, 32)
except:
    font = None
np.set_printoptions(precision=4)


def make_char(text):
    # 创建图像
    image_width = 32
    image_height = 32
    background_color = (255, 255, 255)  # 白色背景
    text_color = (0, 0, 0)  # 黑色文本
    image = Image.new("RGB", (image_width, image_height), background_color)
    draw = ImageDraw.Draw(image)
    # 计算文本位置居中
    text_width, text_height = draw.textsize(text, font=font)
    x = (image_width - text_width) // 2
    y = (image_height - text_height) // 2

    # 绘制文本
    draw.text((x, y), text, font=font, fill=text_color)
    # # 保存图像
    # image.save("output.png")
    return image


def remove_whitespace(image, input_corp):

    x1, y1, w, h = input_corp
    box = (x1, y1, x1 + w, y1 + h)
    cropped_image = image.crop(box)

    image_data = cropped_image.load()

    width, height = cropped_image.size

    top = 0
    bottom = height - 1
    left = 0
    right = width - 1

    while top < height and all(image_data[x, top] == (255, 255, 255) for x in range(width)):
        top += 1

    while bottom >= 0 and all(image_data[x, bottom] == (255, 255, 255) for x in range(width)):
        bottom -= 1

    while left < width and all(image_data[left, y] == (255, 255, 255) for y in range(height)):
        left += 1

    while right >= 0 and all(image_data[right, y] == (255, 255, 255) for y in range(height)):
        right -= 1

    x_original = x1 + left
    y_original = y1 + top
    w_original = right - left + 1
    h_original = bottom - top + 1
    if h_original < 0 or w_original < 0:
        return None
    corp = [x_original, y_original, x_original + w_original, y_original + h_original]
    return corp


def drow_img(image_path,result, save_image_path="res2.jpg"):
    img = cv2.imread(image_path)

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
    cv2.imwrite(save_image_path,img)


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