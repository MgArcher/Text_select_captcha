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
import cv2
from typing import List, Tuple

def find_overall_index_fast(matrix: List[List[float]]) -> List[Tuple[int, int]]:
    """贪心算法寻找全局最优解"""
    if not matrix:
        return []

    mat = np.array(matrix, dtype=np.float64)
    n_rows, n_cols = mat.shape
    k = min(n_rows, n_cols)

    # 存放结果
    index = []

    for _ in range(k):
        # 找到当前全局最大值的扁平化索引
        flat_idx = np.argmax(mat)
        # 转换为二维行列坐标
        row, col = divmod(flat_idx, n_cols)

        index.append((row, col))

        # 将该行和该列的所有元素设为负无穷，禁止再被选中
        mat[row, :] = -np.inf
        mat[:, col] = -np.inf

    # 按行排序（与原逻辑一致）
    index.sort(key=lambda x: x[0])
    return index


def open_image(file, flags=cv2.IMREAD_COLOR):
    """
    使用 OpenCV 读取图像，支持中文路径、numpy数组、bytes。

    Args:
        file: 输入，可以是文件路径（str 或 Path）、numpy 数组、bytes 数据
        flags: cv2.imdecode 的标志，默认为彩色（cv2.IMREAD_COLOR）

    Returns:
        np.ndarray: OpenCV 格式的图像（BGR 通道）
    """
    if isinstance(file, np.ndarray):
        # 已经是 numpy 数组，直接返回（假设其为合法图像）
        return file
    elif isinstance(file, bytes):
        # 从 bytes 数据解码
        data = np.frombuffer(file, dtype=np.uint8)
        img = cv2.imdecode(data, flags)
        return img
    else:
        # 文件路径（字符串或 Path 对象），以二进制方式读取，避免中文路径问题
        path = str(file)
        with open(path, 'rb') as f:
            data = np.frombuffer(f.read(), dtype=np.uint8)
        img = cv2.imdecode(data, flags)
        return img