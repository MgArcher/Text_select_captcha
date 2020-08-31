#!/usr/bin/env python

# -*- coding: utf-8 -*-

"""
@author: jiajia
@file: tool.py
@time: 2020/8/31 9:14
"""
import matplotlib.pyplot as plt

plt.rcParams['font.sans-serif'] = ['SimHei'] # 用来正常显示中文标签
plt.rcParams['axes.unicode_minus'] = False # 用来正常显示负号


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
    image_ = plt.imread(img_path)
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


