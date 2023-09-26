# !/usr/bin/env python
# -*-coding:utf-8 -*-

"""
# File       : b.py
# Time       ：2023/9/21 14:46
# Author     ：yujia
# version    ：python 3.6
# Description：
"""
import numpy as np


def q(matrix):
    rows, cols = matrix.shape
    for row in range(rows):
        if np.all(matrix[row] == matrix[row][0]):
            return True
    for col in range(cols):
        if np.all(matrix[:,col] == matrix[:,col][0]):
            return True
    return False


def fm(matrix, rows, cols, row, col, i):
    if row != 0 and i == 1:
        # 上
        matrix[row][col], matrix[row - 1][col] = matrix[row - 1][col], matrix[row][col]
    if row != rows - 1 and i == 2:
        # 下
        matrix[row][col], matrix[row + 1][col] = matrix[row + 1][col], matrix[row][col]
    if col != 0 and i == 3:
        # 左
        matrix[row][col], matrix[row][col - 1] = matrix[row][col - 1], matrix[row][col]
    if col != cols - 1 and i == 4:
        # 右
        matrix[row][col], matrix[row][col + 1] = matrix[row][col + 1], matrix[row][col]
    return matrix


def md(p, y):
    if y == 1:
        p[0] = p[0] - 1
    elif y == 2:
        p[0] = p[0] + 1
    elif y == 3:
        p[1] = p[1] - 1
    elif y == 4:
        p[1] = p[1] + 1
    return p


def mk(p, y):
    if y == 1:
        p = p - 3
    elif y == 2:
        p = p + 3
    elif y == 3:
        p = p - 1
    elif y == 4:
        p = p + 1
    return p


def m(matr,rows, cols, row, col):
    for i in range(1, 5):
        matrix = matr.copy()
        x = fm(matrix, rows, cols, row, col, i)
        if q(x) is True:
            return True, i
    else:
        return False, 0


def n(l):
    matr = np.array(l).T
    # print(matr)
    k = ["上", "下", "左", "右"]
    rows, cols = matr.shape
    n = 1
    for row in range(rows):
        for col in range(cols):
            x, y = m(matr,rows, cols, row, col)
            if x:
                print(f"{[row, col]}----->{md([row, col], y)}")
                print(f"{n}----->{mk(n, y)}")
                print(k[y - 1])
                # break
                # print(matr[2][1], matr[1][1])
                return x
            n = n + 1


    return




if __name__ == '__main__':
    l = [[0, 2, 3], [1, 1, 3], [3, 2, 1]]
    # l = [[2, 1, 2], [3, 0, 2], [2, 3, 3]]
    # l = [[1, 3, 2], [2, 1, 0], [2, 1, 1]]
    # l = [[2, 1, 0], [2, 0, 2], [0, 0, 1]]
    l = [[1, 2, 1], [3, 1, 2], [2, 0, 3]]
    n(l)
