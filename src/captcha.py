# !/usr/bin/env python
# -*-coding:utf-8 -*-
"""
@author: jiajia
@file: captcha.py
@time: 2021/3/28 15:31
"""
from src.utils.operation import CRNN, CNN, YOLO
from src.utils.discern import Text


import os, sys
root_path = os.getcwd()
# print(root_path)
sys.path.insert(0, root_path + r"\src\utils") # ModuleNotFoundError: No module named 'models'


class TextSelectCaptcha(object):
    def __init__(self, GPU=False, char_dict="ch_sim_char_7255.txt", cnn_path="cnn.onnx", crnn_path="crnn.onnx", yolo_path="yolo.onnx"):
        save_path = os.path.join(os.path.dirname(__file__), 'model')

        path = lambda a, b: os.path.join(a, b)
        char_path = path(save_path, char_dict)

        self.crnn = CRNN(path(save_path, crnn_path), char_path)
        self.cnn = CNN(path(save_path, cnn_path), char_path)
        self.text = Text(self.cnn, self.crnn)
        self.yolo = YOLO(path(save_path, yolo_path))
        
    def run(self, img):
        """
        检测识别
        :param img: 图片的路径、二进制数据或图片矩阵
        :return: list ---> [{"classes": "", "content": "", "crop": []}]
        """
        res = self.detection(img)
        results = self.discern(img, res)
        return results

    def detection(self, img):
        """
        检测
        :param img: 图片的路径、二进制数据或图片矩阵
        :return: list ---> [{'crop': [x1, y1, x2, y2], 'classes': ''}
        """
        res = self.yolo.decect(img)
        return res

    def text_discern(self, img):
        """
        长文本检测
        :param img: 图片的路径、二进制数据或图片矩阵
        :return: str
        """
        return self.crnn.decect(img)
    
    def single_discern(self, img):
        """
        单文本检测
        :param img: 图片的路径、二进制数据或图片矩阵
        :return: str
        """
        return self.cnn.decect(img)

    def discern(self, img, res):
        """
        文本检测
        :param img: 图片的路径、二进制数据或图片矩阵
        :param res:
        :return: list ---> [{"classes": "", "content": "", "crop": []}]
        """
        return self.text.text_predict(res, img)