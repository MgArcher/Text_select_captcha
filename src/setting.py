#!/usr/bin/env python

# -*- coding: utf-8 -*-

"""
@author: jiajia
@file: setting.py
@time: 2020/8/18 9:35
"""
class OPT(object):
    GPU = False
    TEXT_NAME = "config/text.jpg"
    with open("config/characters.txt", 'r', encoding='utf-8') as fr:
        CHARACTERS = fr.read().replace('\n', "")

"""预测模块参数"""
class yolo_opt(OPT):
    model_def = "config/yolov3-tiny.cfg"
    class_path = "config/classes.names"
    weights_path = "model/yolov3_ckpt.pth"

    # 一些参数
    img_size = 416
    conf_thres = 0.8
    nms_thres = 0.4
    batch_size = 1
    n_cpu = 0


"""crnn识别标题模块"""
class crnn_opt(OPT):
    LSTMFLAG = True
    ocrModel = "model/ocr-lstm.pth"
    alphabet = OPT.CHARACTERS
    nclass = len(alphabet) + 1


"""cnn文字识别模块参数"""
class cnn_opt(OPT):
    # 加载模型
    model_path = "model/cnn_iter.pth"

    WIDTH = 64
    HEIGHT = 64
    N_CLASS = len(OPT.CHARACTERS)


"""语序模型"""
class kenlm_opt():
    model_path = 'model/people_chars_lm.klm'