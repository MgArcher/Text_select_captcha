#!/usr/bin/env python

# -*- coding: utf-8 -*-

"""
@author: jiajia
@file: demo.py
@time: 2020/8/13 13:44
"""
import cv2
import random
import time
from PIL import Image
import matplotlib.pyplot as plt

from src import orientation
from src import discern
from src import word_order
# plt.rcParams['font.family'] = ['STFangsong']
plt.rcParams['font.sans-serif'] = ['SimHei'] #用来正常显示中文标签
plt.rcParams['axes.unicode_minus'] = False #用来正常显示负号


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
    image_ = plt.imread(img_path)
    # matrix = numpy.asarray(image)

    # plt.imshow(image_, interpolation='none')
    current_axis = plt.gca()
    for box_ in data:
        box = box_['crop']
        x1, y1, x2, y2 = box
        box_w = x2 - x1
        box_h = y2 - y1

        if box_['classes'] == "target":
            m = box_['content']
            im = image_[y1:y2, x1:x2]
            try:
                cv2.imencode('.jpg', im)[1].tofile(
                    'data/text_image/' + str(random.randint(1000, 9999)) + '_' + m + '.jpg')
            except:
                pass
        if box_['classes'] == "title":
            m = box_['content']
            im = image_[y1:y2, x1:x2]
            try:
                cv2.imencode('.jpg', im)[1].tofile(
                    'data/text_title/' + str(random.randint(1000, 9999)) + '_' + m + '.jpg')
            except:
                pass
        current_axis.add_patch(
            plt.Rectangle((x1, y1), box_w, box_h, color='blue', fill=False, linewidth=2))
    #     plt.text(
    #         x1,
    #         y1,
    #         s=box_['content'],
    #         color="white",
    #         verticalalignment="top",
    #         bbox={"color": "black", "pad": 0},
    #     )
    # plt.show()


def run_click(path):
    """方式一"""
    return discern.text_predict(orientation.location_predict(path), path)


def run_word_order(path):
    """方式二"""
    return word_order.text_predict(orientation.location_predict(path), path)


if __name__ == "__main__":
    res = run_click("bilbil.jpg")
    # path = "test/img_4011.jpg"
    # start = time.time()
    # res = run_click(path)
    # res = run_word_order(path)
    # print(res)
    # print("识别耗时为：", time.time() - start)
    # draw(path, res)

    # res = run_word_order(path)
    # print(res)
    # print("识别耗时为：", time.time() - start)
    # draw(path, res)
    # #
    # import os
    # n = 0
    # for f in os.scandir(r'error'):
    # # for f in os.scandir(r'test'):
    #     print(f.path)
    #     path = f.path
    #     s = time.time()
    #     res = run_click(path)
    #     draw(path, res)
    #     # res = run_word_order(path)
    #     # draw(path, res)
    #     print(res)
    #     m = [i for i in res if i['classes'] == "title"]
    #     print(m)
    #     print("识别耗时为：", time.time() - s)
    #     # draw(path, res)
    #     n = n + 1
    #     if n > 10:
    #         break





