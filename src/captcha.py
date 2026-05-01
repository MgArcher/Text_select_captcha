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
from typing import List, Tuple, Dict, Any

from src.utils import ver_onnx
from src.utils import yolo_onnx
from src.utils import matchingMode


class TextSelectCaptcha(object):
    def __init__(self, per_path: str = 'pre_model_v7.onnx', yolo_path: str = 'best_v3.onnx') -> None:
        """
        初始化识别器，加载 ONNX 模型。

        :param per_path: 字符匹配模型文件名（PreONNX），相对路径，默认 'pre_model_v6.onnx'
        :param yolo_path: YOLO 目标检测模型文件名，默认 'best_v3.onnx'
        """
        save_path = os.path.join(os.path.dirname(__file__), '../model')
        path = lambda a, b: os.path.join(a, b)
        per_path = path(save_path, per_path)
        yolo_path = path(save_path, yolo_path)
        self.yolo = yolo_onnx.YOLO(yolo_path)
        self.pre = ver_onnx.PreONNX(per_path)

    def run(self, image_path: str) -> List[List[float]]:
        img = matchingMode.open_image(image_path)
        data = self.yolo.inference(img)
        target_boxes = [item[:4] for item in data if len(item) >= 6 and item[5] == 0]
        char_boxes = [item[:4] for item in data if len(item) >= 6 and item[5] == 2]
        char_boxes.sort(key=lambda box: box[0])
        img_targets = [img[int(box[1]):int(box[3]), int(box[0]):int(box[2])] for box in target_boxes]
        chars = [img[int(box[1]):int(box[3]), int(box[0]):int(box[2])] for box in char_boxes]
        slys = self.pre.reason_all_batch(chars, img_targets)
        sorted_result = matchingMode.find_overall_index_fast(slys)
        result = [target_boxes[j] for i, j in sorted_result]
        return result

    def run_dict(self, image_path: str) -> Dict[str, Any]:
        img = matchingMode.open_image(image_path)
        h, w, _ = img.shape
        result = self.run(image_path)
        return {
            "imgW": w,
            "imgH": h,
            "point": [{"x_rel": (x1 + x2) / 2, "y_rel": (y1 + y2) / 2} for x1, y1, x2, y2 in result],
            "corp": [{"x1": x1, "y1": y1 , "x2": x2, "y2":  y2 } for x1, y1, x2, y2 in result],
        }


if __name__ == '__main__':
    from src.drawing import drow_img
    cap = TextSelectCaptcha()
    image_path = r"../docs/res.jpg"
    image_path = r"D:\captcha/5cf82cae41fc4f18bd257f96dece6e34.jpg"
    result = cap.run(image_path)
    print(result)
    drow_img(image_path, result)
    print(cap.run_dict(image_path))