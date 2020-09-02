#!/usr/bin/env python

# -*- coding: utf-8 -*-

"""
@author: jiajia
@file: demo2.py
@time: 2020/8/17 16:00
"""
import torch
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
from torchvision.transforms.functional import to_tensor

from src.utils.cnn_model import ConvNet
from src.utils.network_torch import CRNN
from src.setting import crnn_opt
from src.setting import cnn_opt




"""加载模型"""
"""cnn图形文字识别"""
if cnn_opt.GPU:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
else:
    device = torch.device("cpu")
net = ConvNet(cnn_opt.N_CLASS).to(device)
net.load_state_dict(torch.load(cnn_opt.model_path, map_location="cuda:0" if torch.cuda.is_available() and cnn_opt.GPU else "cpu"))
net.eval()

"""crnn  标题文本识别"""
crnn = CRNN(
    32,
    1,
    crnn_opt.nclass,
    256,
    leakyRelu=False,
    lstmFlag=crnn_opt.LSTMFLAG,
    GPU=crnn_opt.GPU,
    alphabet=crnn_opt.alphabet
)

crnn.load_weights(crnn_opt.ocrModel)
crnn.eval()

"""加载模型"""

def update_text(text_list, title):
    res = [i[0] for i in text_list]
    # 判断是否有重复的文字
    if len(res) != len(set(res)):
        # 判断标题长度与框出来的文字是否长度相同
        if len(text_list) == len(title):
            # 对出现了相同字符的字，把出现概率低的替换
            no_text = set(title) - set([i[0] for i in text_list])
            S = set()
            W = set()
            for i in range(len(text_list)):
                if text_list[i][0] in S:
                    W.add(text_list[i][0])
                S.add(text_list[i][0])
            for w in W:
                qu = [i for i in text_list if i[0] == w]
                u = max(qu, key=lambda x: x[1])
                for i in qu:
                    if i != u:
                        number = text_list.index(i)
                        # 随机给一个
                        text_list[number] = (no_text.pop())

            results = [i[0] for i in text_list]
        else:
            results = [i[0] for i in text_list]
    else:
        results = res
    return results


def get_text(X, title):
    # 获得每个字的可能概率
    if torch.cuda.is_available() and cnn_opt.GPU:
        X = X.cuda()
    outputs = net(X)
    if cnn_opt.GPU:
        outputs = outputs.cuda().data.cpu()
    text_list = []
    # 获取标签所在的字符位置
    title_Y = [cnn_opt.CHARACTERS.find(i) for i in title]
    # 选取标题中字符中概率最高的
    for i in outputs:
        y = []
        for j in title_Y:
            y.append(i[j])
        # 获得其中的最大值
        if y:
            y = np.mat(y)
            text_list.append((title[np.argmax(y)], y.max()))
        else:
            text_list.append((cnn_opt.CHARACTERS[torch.argmax(i)], float(i.max())))
    # 修改结果
    results = update_text(text_list, title)
    return results


def text_predict(res, image_path):
    """文本预测"""
    image_ = plt.imread(image_path)
    zreo = lambda p: [0 if x < 0 else x for x in p]
    X = torch.zeros((len([i for i in res if i.get("classes") == "target"]), 3, cnn_opt.HEIGHT, cnn_opt.WIDTH))
    x_number = 0
    title = ""
    sign = False
    for i, r in enumerate(res):
        classes = r.get("classes")
        crop = r.get("crop")
        crop = zreo(crop)
        x1, y1, x2, y2 = crop

        if classes == "target":
            im = image_[y1:y2, x1:x2]
            out = Image.fromarray(im)
            out = out.resize((cnn_opt.WIDTH, cnn_opt.HEIGHT), Image.ANTIALIAS)
            out = out.convert('RGB')
            X[x_number] = to_tensor(out)
            x_number += 1
            sign = True
        if classes == "title":
            im = image_[y1:y2, x1:x2]
            partImg = Image.fromarray(im)
            title = crnn.predict(partImg)

    if sign:
        text_list = get_text(X, title)
        x_number = 0
        for i, r in enumerate(res):
            classes = r.get("classes")
            if classes == "target":
                res[i]['content'] = text_list[x_number]
                x_number += 1
            else:
                res[i]['content'] = title

    return res



