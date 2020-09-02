#!/usr/bin/env python

# -*- coding: utf-8 -*-

"""
@author: jiajia
@file: mode_one.py
@time: 2020/8/31 9:16
"""
from src import orientation
from src import discern
from src.tool import draw


def run_click(path):
    """方式一"""
    return discern.text_predict(orientation.location_predict(path), path)


if __name__ == '__main__':
    import time
    path = "test/b.jpg"
    start = time.time()
    res = run_click(path)
    print(res)
    print("识别耗时为：", time.time() - start)
    draw(path, res)