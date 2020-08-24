#!/usr/bin/env python

# -*- coding: utf-8 -*-

"""
@author: jiajia
@file: demo.py
@time: 2020/8/13 13:44
"""
import time
from PIL import Image
import matplotlib.pyplot as plt

from src.orientation import location_predict
from src.discern import text_predict
plt.rcParams['font.family'] = ['STFangsong']

def to_selenium(res):
    place = []
    title = [i['content'] for i in res if i['classes'] == "title"][0]
    for t in title:
        for item in res:
            if item['classes'] == "target":
                x1, y1, x2, y2 = item['crop']
                if item['content'] == t:
                    place.append(
                        {
                            "text": t,
                            "place": [(x1 + x2)/2, (y1 + y2)/2]
                        }
                    )
    return place

def draw(img_path, data):
    "绘制识别结果"
    image = Image.open(img_path)
    # matrix = numpy.asarray(image)

    plt.imshow(image, interpolation='none')
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


def run(path):
    return text_predict(location_predict(path), path)

if __name__ == "__main__":
    path = "test/img_2.jpg"

    start = time.time()
    res = run(path)
    print(res)
    print("识别耗时为：", time.time() - start)
    draw(path, res)

    # import os
    # for f in os.scandir('test'):
    #     print(f.path)
    #     path = f.path
    #     s = time.time()
    #     res = text_predict(location_predict(path), path)
    #     print(res)
    #     print("识别耗时为：", time.time() - s)
    #     draw(path, res)





