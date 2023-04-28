# !/usr/bin/env python
# -*-coding:utf-8 -*-

"""
# File       : a.py
# Time       ：2023/4/27 9:41
# Author     ：yujia
# version    ：python 3.6
# Description：
"""

import nltk

# 定义顺序限定词性元组，包含三个元素分别代表：词性、词性所在位置、排序位置
pos_order = [('n', 1, 0), ('nr', 1, 1), ('ns', 1, 1), ('nt', 1, 1), ('nz', 1, 1),
             ('a', 1, 1), ('ad', 1, 1), ('v', 2, 2), ('d', 2, 1), ('m', 2, 1),
             ('ad', 2, 1), ('q', 3, 1), ('p', 3, 1), ('c', 3, 2), ('u', 3, 1),
             ('e', 3, 2), ('o', 3, 1), ('i', 3, 1), ('l', 3, 1), ('wp', 3, 2),
             ('ws', 3, 2), ('x', 3, 2), ('uj', 3, 1), ('ul', 3, 1), ('vg', 3, 2),
             ('eng', 3, 1)]

def get_next_word(words, index, pos):
    """
    从指定位置开始，获取紧跟其后的下一个符合特定词性的单词
    """
    while index < len(words):
        w, p = words[index]
        if p == pos:
            return w, index
        index += 1
    return None, len(words)

def get_order_pos(pos):
    """
    获取指定词性在排序规则中的位置
    """
    for p in pos_order:
        if p[0] == pos:
            return p[2]
    return -1

def semantic_sort(words):
    """
    使用nltk对单词按照语义排序并返回
    """
    if not words:
        return ''
    tagged_words = nltk.pos_tag(words)  # 对单词进行词性标注
    order_words = []  # 存放排序后的单词和词性
    for w, p in tagged_words:
        order_pos = get_order_pos(p)  # 获取词性在排序规则中的位置
        if order_pos >= 0:
            order_words.append((w, p, order_pos))  # 添加排序后的单词和词性
    # 根据词性在排序规则中的位置进行排序
    order_words = sorted(order_words, key=lambda x: (x[2],))
    result = ''
    index = 0
    # 按照排序后的顺序依次获取单词
    while index < len(order_words):
        w, index = get_next_word(order_words, index, 1)
        if w:
            result += w
    return result



semantic_sort([('记', 'v'), ('西', 'ns'), ('游', 'n')])

# text = "西游记"
# print(semantic_sort(text))
# text = "游西记"
# print(semantic_sort(text))