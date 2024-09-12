# !/usr/bin/env python
# -*-coding:utf-8 -*-

"""
# File       : jy_click.py
# Time       ：2023/11/13 16:49
# Author     ：yujia
# version    ：python 3.6
# Description：
"""
import os

from src.utils import ver_onnx
from src.utils import yolo_onnx
from src.utils import utils, matchingMode


class JYClick(object):
    def __init__(self, per_path='pre_model_v3.bin', yolo_path='best_v2.bin', sign=True):
        """
        jiyan 最好 pre_model_v3.onnx
        nine 最好  pre_model_v5.onnx

        """
        save_path = os.path.join(os.path.dirname(__file__), '../../model')
        path = lambda a, b: os.path.join(a, b)
        per_path = path(save_path, per_path)
        yolo_path = path(save_path, yolo_path)
        if sign:
            try:
                from src.utils.load import decryption
            except:
                raise Exception("Error! 请在windows下的python3.6、3.8、3.10环境下使用")
            yolo_path = decryption(yolo_path)
            per_path = decryption(per_path)
        self.yolo = yolo_onnx.YOLOV5_ONNX(yolo_path, classes=['target', 'title', 'char'], providers=['CPUExecutionProvider'])
        self.pre = ver_onnx.PreONNX(per_path, providers=['CPUExecutionProvider'])

    def run(self, image_path):
        img = utils.open_image(image_path)
        data = self.yolo.decect(image_path)
        # 需要选择的字
        targets = [i.get("crop") for i in data if i.get("classes") == "target"]
        chars = [i.get("crop") for i in data if i.get("classes") == "char"]
        # 根据坐标进行排序
        chars.sort(key=lambda x: x[0])
        chars = [img.crop(char) for char in chars]
        img_targets = [img.crop(target) for target in targets]
        slys = [self.pre.reason_all(img_char, img_targets) for img_char in chars]
        sorted_result = matchingMode.find_overall_index(slys)
        result = [targets[j] for i, j in sorted_result]
        return result


if __name__ == '__main__':
    image_path = "../../docs/res.jpg"
    cap = JYClick()
    result = cap.run(image_path)
    print(result)
    utils.drow_img(image_path, result, "jy_click.jpg")



