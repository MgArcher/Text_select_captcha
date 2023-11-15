# !/usr/bin/env python
# -*-coding:utf-8 -*-

"""
# File       : ver_onnx.py.py
# Time       ：2023/3/29 14:20
# Author     ：yujia
# version    ：python 3.6
# Description：
"""
from PIL import Image
import onnxruntime
import numpy as np
import time
import numbers
from io import BytesIO

import cv2
import random


np.set_printoptions(precision=4)


def drow_img(image_path,result, save_image_path="res2.jpg"):
    img = cv2.imread(image_path)

    def plot_one_box(x, img, color=None, label=None, line_thickness=None):
        # Plots one bounding box on image img
        tl = line_thickness or round(0.002 * (img.shape[0] + img.shape[1]) / 2) + 1  # line/font thickness
        color = color or [random.randint(0, 255) for _ in range(3)]
        c1, c2 = (int(x[0]), int(x[1])), (int(x[2]), int(x[3]))
        cv2.rectangle(img, c1, c2, color, thickness=tl, lineType=cv2.LINE_AA)
        if label:
            tf = max(tl - 1, 1)  # font thickness
            t_size = cv2.getTextSize(label, 0, fontScale=tl / 3, thickness=tf)[0]
            c2 = c1[0] + t_size[0], c1[1] - t_size[1] - 3
            cv2.rectangle(img, c1, c2, color, -1, cv2.LINE_AA)  # filled
            cv2.putText(img, label, (c1[0], c1[1] - 2), 0, tl / 3, [225, 255, 255], thickness=tf, lineType=cv2.LINE_AA)
        return img

    colors = [[random.randint(0, 255) for _ in range(3)] for _ in range(len(result))]
    for i, xyxy in enumerate(result):
        label = i + 1
        img = plot_one_box(xyxy, img, label=str(label), color=colors[i], line_thickness=1)
    cv2.imwrite(save_image_path,img)


def preprocess_input(x):
    x /= 255.0
    return x


def crop(img, i, j, h, w):
    """Crop the given PIL Image.

    Args:
        img (PIL Image): Image to be cropped.
        i (int): i in (i,j) i.e coordinates of the upper left corner.
        j (int): j in (i,j) i.e coordinates of the upper left corner.
        h (int): Height of the cropped image.
        w (int): Width of the cropped image.

    Returns:
        PIL Image: Cropped image.
    """
    return img.crop((j, i, j + w, i + h))


def center_crop(img, output_size):
    if isinstance(output_size, numbers.Number):
        output_size = (int(output_size), int(output_size))
    w, h = img.size
    th, tw = output_size
    i = int(round((h - th) / 2.))
    j = int(round((w - tw) / 2.))
    return crop(img, i, j, th, tw)


def resize(img, size, interpolation=Image.BILINEAR):
    r"""Resize the input PIL Image to the given size.

    Args:
        img (PIL Image): Image to be resized.
        size (sequence or int): Desired output size. If size is a sequence like
            (h, w), the output size will be matched to this. If size is an int,
            the smaller edge of the image will be matched to this number maintaing
            the aspect ratio. i.e, if height > width, then image will be rescaled to
            :math:`\left(\text{size} \times \frac{\text{height}}{\text{width}}, \text{size}\right)`
        interpolation (int, optional): Desired interpolation. Default is
            ``PIL.Image.BILINEAR``

    Returns:
        PIL Image: Resized image.
    """
    if isinstance(size, int):
        w, h = img.size
        if (w <= h and w == size) or (h <= w and h == size):
            return img
        if w < h:
            ow = size
            oh = int(size * h / w)
            return img.resize((ow, oh), interpolation)
        else:
            oh = size
            ow = int(size * w / h)
            return img.resize((ow, oh), interpolation)
    else:
        return img.resize(size[::-1], interpolation)


def cvtColor(image):
    if len(np.shape(image)) == 3 and np.shape(image)[2] == 3:
        return image
    else:
        image = image.convert('RGB')
        return image


def letterbox_image(image, size, letterbox_image):
    w, h = size
    iw, ih = image.size
    if letterbox_image:
        '''resize image with unchanged aspect ratio using padding'''
        scale = min(w/iw, h/ih)
        nw = int(iw*scale)
        nh = int(ih*scale)

        image = image.resize((nw,nh), Image.BICUBIC)
        new_image = Image.new('RGB', size, (128,128,128))
        new_image.paste(image, ((w-nw)//2, (h-nh)//2))
    else:
        if h == w:
            new_image = resize(image, h)
        else:
            new_image = resize(image, [h ,w])
        new_image = center_crop(new_image, [h ,w])
    return new_image


class PreONNX(object):
    def __init__(self, path, providers=None):
        if not providers:
            providers = ['CPUExecutionProvider']
        self.sess = onnxruntime.InferenceSession(path, providers=providers)
        self.loadSize = 512
        self.input_shape = [105, 105]

    def sigmoid(self, x):
        return 1 / (1 + np.exp(-x))

    def zhuanhuan(self, file):
        # 图片转换为矩阵
        if isinstance(file, np.ndarray):
            img = Image.fromarray(file)
        elif isinstance(file, bytes):
            img = Image.open(BytesIO(file))
        elif isinstance(file, Image.Image):
            img = file
        else:
            img = Image.open(file)
        return img

    def open_image(self, file, input_shape, nc=3):
        out = self.zhuanhuan(file)
        # 改变大小 并保证其不失真
        out = out.convert('RGB')
        h, w = input_shape
        out = out.resize((w, h), 1)
        if nc == 1:
            out = out.convert('L')
        return out

    def set_img(self, lines):
        image = self.open_image(lines, self.input_shape, 3)
        image = np.array(image).astype(np.float32) / 255.0
        photo = np.expand_dims(np.transpose(image, (2, 0, 1)), 0)
        return photo

    def reason(self, image_1, image_2):
        photo_1 = self.set_img(image_1)
        photo_2 = self.set_img(image_2)
        out = self.sess.run(None, {"x1": photo_1, "x2": photo_2})
        out = out[0]
        out = self.sigmoid(out)
        out = out[0][0]
        return out

    def reason_all(self, image_1, image_2_list):
        photo_1 = self.set_img(image_1)
        photo_2_all = None
        photo_1_all = photo_1
        for image_2 in image_2_list:
            photo_2 = self.set_img(image_2)
            if photo_2_all is None:
                photo_2_all = photo_2
            else:
                photo_2_all = np.concatenate((photo_2_all, photo_2))
                photo_1_all = np.concatenate((photo_1_all, photo_1))

        out = self.sess.run(None, {"x1": photo_1_all, "x2": photo_2_all})
        out = out[0]
        out = self.sigmoid(out)
        out = out.tolist()
        out = [i[0] for i in out]

        return out


if __name__ == '__main__':
    pre_onnx_path = "pre_model.onnx"
    pre = PreONNX(pre_onnx_path, providers=['CPUExecutionProvider'])
    image_1 = r"datasets\bilbil\character2\char_1.jpg"
    image_2 = r"datasets\bilbil\character2\plan_4.jpg"
    image_1 = "img.png"
    image_2 = "img_1.png"
    large_img = pre.reason_all(image_1, image_2)
    print(large_img)