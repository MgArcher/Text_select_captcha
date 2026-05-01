# !/usr/bin/env python
# -*-coding:utf-8 -*-

"""
# File       : zz_onnx.py
# Time       ：2022/10/21 17:28
# Author     ：yujia
# version    ：python 3.6
# Description：
"""
import cv2
import numpy as np
import onnxruntime as ort
from typing import List, Tuple, Union, Any


class YOLO:
    def __init__(self, model_path: str, conf_threshold: float = 0.3) -> None:
        """
        初始化 YOLO26 ONNX 模型。

        Args:
            model_path (str): ONNX 模型文件路径。
            conf_threshold (float): 置信度阈值。
        """
        # 使用 ONNX Runtime 创建推理会话
        # 若需 GPU 加速，可添加 providers=['CUDAExecutionProvider', 'CPUExecutionProvider']
        self.session = ort.InferenceSession(model_path, providers=['CPUExecutionProvider'])
        self.conf_threshold = conf_threshold

        # 获取模型输入输出信息
        self.input_name = self.session.get_inputs()[0].name
        self.input_shape = self.session.get_inputs()[0].shape  # 应为 [1, 3, 640, 640]
        self.output_name = self.session.get_outputs()[0].name

    def letterbox(self, image: np.ndarray, target_size: Tuple[int, int] = (640, 640)) -> Tuple[np.ndarray, float, Tuple[int, int, int, int]]:
        """
        保持宽高比对图像进行缩放和填充 (Letterbox)。

        Args:
            image (np.ndarray): 输入图像 (BGR 格式)。
            target_size (tuple): 目标尺寸 (width, height)。

        Returns:
            tuple: (填充后的图像, 缩放比例, 填充的宽高信息)
        """
        img_h, img_w = image.shape[:2]
        target_w, target_h = target_size

        # 计算缩放比例 (选择较小的比例以确保图像能完整放入)
        scale = min(target_w / img_w, target_h / img_h)
        new_w, new_h = int(img_w * scale), int(img_h * scale)

        # 调整图像尺寸
        resized_img = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_LINEAR)

        # 创建画布并填充
        canvas = np.full((target_h, target_w, 3), 114, dtype=np.uint8)
        # 计算填充偏移量 (居中对齐)
        dw, dh = (target_w - new_w) // 2, (target_h - new_h) // 2
        canvas[dh:dh+new_h, dw:dw+new_w, :] = resized_img

        return canvas, scale, (dw, dh, new_w, new_h)

    def preprocess(self, image: np.ndarray) -> Tuple[np.ndarray, float, Tuple[int, int, int, int]]:
        """
        图像预处理。
        1. BGR -> RGB 通道转换。
        2. Letterbox 调整。
        3. 归一化 (0-255 -> 0.0-1.0)。
        4. HWC -> NCHW 格式转换。

        Args:
            image (np.ndarray): 输入图像 (BGR 格式)。

        Returns:
            tuple: (预处理后的数组, 缩放比例, 填充信息)
        """
        # # BGR 转 RGB
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        # Letterbox 填充
        padded_img, scale, pad_info = self.letterbox(image_rgb, target_size=(self.input_shape[3], self.input_shape[2]))

        # 归一化并转换维度 (HWC -> CHW -> NCHW)
        padded_img = padded_img.astype(np.float32) / 255.0
        input_tensor = np.transpose(padded_img, (2, 0, 1))  # CHW
        input_tensor = np.expand_dims(input_tensor, axis=0) # NCHW

        return input_tensor, scale, pad_info

    def inference(self, image: np.ndarray) -> List[List[Union[int, float]]]:
        """
        执行 ONNX 推理和后处理。

        Args:
            image (np.ndarray): 输入图像 (BGR 格式)。

        Returns:
            list: 检测结果列表，每个元素为 [x1, y1, x2, y2, conf, class_id]。
        """
        # 1. 预处理
        input_tensor, scale, (dw, dh, new_w, new_h) = self.preprocess(image)

        # 2. ONNX 推理
        # 输出形状: (N, 300, 6) -> (1, 300, 6)
        outputs = self.session.run([self.output_name], {self.input_name: input_tensor})[0]

        # 3. 后处理: 置信度过滤 + 坐标转换
        detections = []
        for detection in outputs[0]: # 遍历 300 个预测
            x1, y1, x2, y2, conf, class_id = detection.tolist()
            if conf < self.conf_threshold:
                continue

            # 将坐标从填充后的图像空间映射回原始图像空间
            # 移除 padding 并还原缩放
            x1 = max(0, (x1 - dw) / scale)
            y1 = max(0, (y1 - dh) / scale)
            x2 = min(image.shape[1], (x2 - dw) / scale)
            y2 = min(image.shape[0], (y2 - dh) / scale)

            detections.append([int(x1), int(y1), int(x2), int(y2), conf, int(class_id)])

        return detections


