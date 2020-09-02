#!/usr/bin/env python

# -*- coding: utf-8 -*-

"""
@author: jiajia
@file: kens.py
@time: 2020/8/26 11:19
"""
"""
语序验证码识别
"""
from itertools import permutations
import kenlm
import torch
from PIL import Image
import matplotlib.pyplot as plt
from torchvision.transforms.functional import to_tensor

from src.utils.cnn_model import ConvNet
from src.setting import kenlm_opt
from src.setting import cnn_opt

"""加载模型"""
"""cnn图形文字识别"""
if cnn_opt.GPU:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
else:
    device = torch.device("cpu")
# net = ConvNet(cnn_opt.N_CLASS).to(device)
net = ConvNet(cnn_opt.N_CLASS).to(device)
net.load_state_dict(torch.load(cnn_opt.model_path, map_location="cuda:0" if torch.cuda.is_available() and cnn_opt.GPU else "cpu"))
net.eval()
"""加载kenlm模型"""
model = kenlm.LanguageModel(kenlm_opt.model_path)


def n_gram(word):
    word_list = list(word)
    # n-gram
    candidate_list = list(permutations(word_list, r=len(word_list)))
    a_c_s = -100
    a_c = ""
    b_c_s = 1000
    b_c = ""
    for candidate in candidate_list:
        candidate = ' '.join(candidate)
        # print(candidate)
        a = model.score(candidate)
        b = model.perplexity(candidate)
        # 选择得分最高的
        if a > a_c_s:
            a_c = candidate
            a_c_s = a

        # 选择困惑度最小的
        if b_c_s > b:
            b_c = candidate
            b_c_s = b

    return a_c.replace(" ", '')


def get_text(X):
    # 获得每个字的可能概率
    if torch.cuda.is_available() and cnn_opt.GPU:
        X = X.cuda()
    outputs = net(X)
    text_list = []
    # 方式一  识别给每个框框中的字符
    for i in outputs:
        text_list.append(cnn_opt.CHARACTERS[torch.argmax(i)])
    return text_list


def text_predict(res, image_path):
    """文本预测"""
    image_ = plt.imread(image_path)
    zreo = lambda p: [0 if x < 0 else x for x in p]
    X = torch.zeros((len([i for i in res if i.get("classes") == "target"]), 3, cnn_opt.HEIGHT, cnn_opt.WIDTH))
    x_number = 0
    title = ""
    sign = False
    for i, r in enumerate(res[:]):
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
        else:
            res.pop(i)

    if sign:
        text_list = get_text(X)
        x_number = 0
        for i, r in enumerate(res[:]):
            classes = r.get("classes")
            if classes == "target":
                res[i]['content'] = text_list[x_number]
                x_number += 1
        # 预测语序
        title = n_gram(''.join(text_list))
        res.append(
            {
                'classes': "title",
                "crop": [0, 345, 116, 385],
                "content": title
            }
        )
    return res



