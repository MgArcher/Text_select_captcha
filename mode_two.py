#!/usr/bin/env python

# -*- coding: utf-8 -*-

"""
@author: jiajia
@file: mode_two.py
@time: 2020/8/31 9:17
"""
from src import orientation
from src import word_order
from src.tool import draw


def run_word_order(path):
    """方式二"""
    return word_order.text_predict(orientation.location_predict(path), path)


if __name__ == '__main__':
    import time
    path = "test/img_4011.jpg"
    start = time.time()
    res = run_word_order(path)
    print(res)
    print("识别耗时为：", time.time() - start)
    draw(path, res)