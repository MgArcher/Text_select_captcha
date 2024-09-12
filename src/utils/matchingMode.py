# !/usr/bin/env python
# -*-coding:utf-8 -*-

"""
# File       : matchingMode.py
# Time       ：2024/8/18 20:10
# Author     ：yujia
# version    ：python 3.6
# Description：
"""

import numpy as np


def find_overall_index(matrix):
    "寻找全局最大值数依次排序"
    # 转换为 NumPy 数组便于处理
    matrix = np.array(matrix)

    # 存储已选择的行和列索引
    selected_rows = set()
    selected_cols = set()

    # 存储结果
    index = []

    # 获取矩阵的形状
    num_rows, num_cols = matrix.shape

    # 查找每行每列的最大值
    for _ in range(min(num_rows, num_cols)):
        # 找到当前未选择的最大值
        max_value = -np.inf
        max_position = (-1, -1)  # 初始化为无效位置

        for i in range(num_rows):
            if i in selected_rows:
                continue
            for j in range(num_cols):
                if j in selected_cols:
                    continue
                if matrix[i, j] > max_value:
                    max_value = matrix[i, j]
                    max_position = (i, j)

                    # 将找到的最大值位置加入结果
        index.append(max_position)
        # 将行和列标记为已选择
        selected_rows.add(max_position[0])
        selected_cols.add(max_position[1])
    sorted_result = sorted(index, key=lambda x: x[0])
    return sorted_result


def find_overall_zero_index(matrix):
    "修改数组结果进行全局搜索"
    slys = np.array(matrix)
    index = []
    for i in range(len(slys)):
        # print(slys)
        # 获取最大值的索引
        max_index_2d = np.unravel_index(np.argmax(slys), slys.shape)
        index.append(max_index_2d)
        # print("转换后的二维索引：", max_index_2d)
        # 将第一行变为0
        slys[max_index_2d[0], :] = 0
        # 将第二列变为0
        slys[:, max_index_2d[1]] = 0
    sorted_result = sorted(index, key=lambda x: x[0])
    return sorted_result


def find_row_index(matrix):
    "按行寻找最大值"
    pass