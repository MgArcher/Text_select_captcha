#!/usr/bin/env python

# -*- coding: utf-8 -*-

"""
@author: jiajia
@file: cnn_model.py
@time: 2020/8/18 19:32
"""
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


def conv_dw(inp, oup, stride):
    return nn.Sequential(
        # dw
        nn.Conv2d(inp, inp, 3, stride, 1, groups=inp, bias=False),
        nn.BatchNorm2d(inp),
        nn.ReLU6(inplace=True),
        # pw-linear
        nn.Conv2d(inp, oup, 1, 1, 0, bias=False),
        nn.BatchNorm2d(oup),
    )


def conv_bn(inp, oup, stride):
    return nn.Sequential(
        nn.Conv2d(inp, oup, 3, stride, 1, bias=False),
        nn.BatchNorm2d(oup),
        nn.ReLU6(inplace=True)
    )


class ConvNet(nn.Module):
    """input shape (64x64x3)"""
    def __init__(self, num_classes):
        super(ConvNet, self).__init__()
        self.conv1 = conv_bn(3, 8, 1)  # 64x64x1
        self.conv2 = conv_bn(8, 16, 1)  # 64x64x16
        self.conv3 = conv_dw(16, 32, 1)  # 64x64x32
        self.conv4 = conv_dw(32, 32, 2)  # 32x32x32
        self.conv5 = conv_dw(32, 64, 1)  # 32x32x64
        self.conv6 = conv_dw(64, 64, 2)  # 16x16x64
        self.conv7 = conv_dw(64, 128, 1)  # 16x16x128
        self.conv8 = conv_dw(128, 128, 1)  # 16x16x128
        self.conv9 = conv_dw(128, 128, 1)  # 16x16x128
        self.conv10 = conv_dw(128, 128, 1)  # 16x16x128
        self.conv11 = conv_dw(128, 128, 1)  # 16x16x128
        self.conv12 = conv_dw(128, 256, 2)  # 8x8x256
        self.conv13 = conv_dw(256, 256, 1)  # 8x8x256
        self.conv14 = conv_dw(256, 256, 1)  # 8x8x256
        self.conv15 = conv_dw(256, 512, 2)  # 4x4x512
        self.conv16 = conv_dw(512, 512, 1)  # 4x4x512
        self.classifier = nn.Sequential(
            nn.Linear(512*4*4, 1024),
            nn.Dropout(0.2),
            nn.ReLU(inplace=True),
            nn.Linear(1024, num_classes),
        )
        self.weight_init()

    def forward(self, x):
        x1 = self.conv1(x)
        x2 = self.conv2(x1)
        x3 = self.conv3(x2)
        x4 = self.conv4(x3)
        x5 = self.conv5(x4)
        x6 = self.conv6(x5)
        x7 = self.conv7(x6)
        x8 = self.conv8(x7)
        x9 = self.conv9(x8)
        x9 = F.relu(x8 + x9)
        x10 = self.conv10(x9)
        x11 = self.conv11(x10)
        x11 = F.relu(x10 + x11)
        x12 = self.conv12(x11)
        x13 = self.conv13(x12)
        x14 = self.conv14(x13)
        x14 = F.relu(x13 + x14)
        x15 = self.conv15(x14)
        x16 = self.conv16(x15)
        x = x16.view(x16.size(0), -1)
        x = self.classifier(x)
        return x

    def weight_init(self):
        for layer in self.modules():
            self._layer_init(layer)

    def _layer_init(self, m):
        # 使用isinstance来判断m属于什么类型
        if isinstance(m, nn.Conv2d):
            n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
            m.weight.data.normal_(0, np.sqrt(2. / n))
        elif isinstance(m, nn.BatchNorm2d):
            # m中的weight，bias其实都是Variable，为了能学习参数以及后向传播
            m.weight.data.fill_(1)
            m.bias.data.zero_()
        elif isinstance(m, nn.Linear):
            m.weight.data.normal_(0, 0.01)
            m.bias.data.zero_()