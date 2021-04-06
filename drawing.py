"""
@author: jiajia
@file: drawing.py
@time: 2021/3/28 15:31
"""
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
from io import BytesIO
plt.rcParams['font.sans-serif'] = ['SimHei'] #用来正常显示中文标签
plt.rcParams['axes.unicode_minus'] = False #用来正常显示负号


def open_image(file):
    if isinstance(file, np.ndarray):
        img = Image.fromarray(file)
    elif isinstance(file, bytes):
        img = Image.open(BytesIO(file))
    else:
        img = Image.open(file)
    img = np.array(img)
    return img


def draw(img_path, data):
    "绘制识别结果"
    image_ = open_image(img_path)

    plt.imshow(image_, interpolation='none')
    current_axis = plt.gca()
    for box_ in data:
        box = box_['crop']
        x1, y1, x2, y2 = box
        box_w = x2 - x1
        box_h = y2 - y1

        current_axis.add_patch(
            plt.Rectangle((x1, y1), box_w, box_h, color='blue', fill=False, linewidth=2))
        plt.text(
            x1,
            y1,
            s=box_['content'],
            color="white",
            verticalalignment="top",
            bbox={"color": "black", "pad": 0},
        )
    plt.show()
