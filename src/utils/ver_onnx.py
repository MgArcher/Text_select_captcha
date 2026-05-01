# !/usr/bin/env python
# -*-coding:utf-8 -*-

"""
# File       : val_onnx.py
# Time       ：2026/4/30 17:54
# Author     ：yujia
# version    ：python 3.6
# Description：
"""
import os
import cv2
import numpy as np
import onnxruntime as ort


def cvtColor(image_np):
    """确保图像为 3 通道 RGB 格式"""
    if len(image_np.shape) == 3 and image_np.shape[2] == 3:
        return cv2.cvtColor(image_np, cv2.COLOR_BGR2RGB)
    elif len(image_np.shape) == 3 and image_np.shape[2] == 4:
        bgr = cv2.cvtColor(image_np, cv2.COLOR_BGRA2BGR)
        return cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    else:
        return cv2.cvtColor(image_np, cv2.COLOR_GRAY2RGB)


def letterbox_image(image_np, target_size):
    h, w = target_size
    ih, iw = image_np.shape[:2]
    scale = min(w / iw, h / ih)
    nw = int(iw * scale)
    nh = int(ih * scale)
    resized = cv2.resize(image_np, (nw, nh), interpolation=cv2.INTER_CUBIC)
    new_image = np.full((h, w, 3), 128, dtype=np.uint8)
    dx = (w - nw) // 2
    dy = (h - nh) // 2
    new_image[dy:dy + nh, dx:dx + nw] = resized
    return new_image


def preprocess_input(x):
    return x.astype(np.float32) / 255.0


def preprocess_image(img: np.ndarray, input_size=(112, 112)) -> np.ndarray:
    """读取图像并预处理，返回形状为 [1, 3, H, W] 的 numpy 数组"""
    img = cvtColor(img)
    img = letterbox_image(img, input_size)
    img = preprocess_input(img)  # 归一化到 [0, 1]
    img = np.transpose(img, (2, 0, 1)).astype(np.float32)  # HWC -> CHW
    return np.expand_dims(img, axis=0)  # [1, 3, H, W]


# ===================== ONNX 推理接口 =====================
class PreONNX:
    def __init__(self, onnx_path: str, device: str = 'cpu', input_size=(112, 112)):
        """
        onnx_path : ONNX 模型文件路径
        device    : 'cpu' 或 'cuda' (需要 onnxruntime-gpu)
        input_size: 输入图像尺寸，需与导出时一致
        """
        self.input_size = input_size

        # 配置推理提供者
        providers = ['CPUExecutionProvider']
        if device == 'cuda':
            providers.insert(0, 'CUDAExecutionProvider')

        self.session = ort.InferenceSession(onnx_path, providers=providers)
        self.input_names = [inp.name for inp in self.session.get_inputs()]
        self.output_names = [out.name for out in self.session.get_outputs()]

    def predict_pair(self, img1:  np.ndarray, img2: np.ndarray) -> float:
        """比较两张图像，返回相似概率 (0~1)"""
        img1 = preprocess_image(img1, self.input_size)
        img2 = preprocess_image(img2, self.input_size)

        # 注意：输入名称需与导出时一致（默认为 'input1', 'input2'）
        ort_inputs = {
            self.input_names[0]: img1,
            self.input_names[1]: img2
        }
        logits = self.session.run(self.output_names, ort_inputs)[0]  # shape: [1, 1]
        prob = 1.0 / (1.0 + np.exp(-logits))  # sigmoid
        return float(prob[0, 0])

    def _reason_all_batch(self, img1_paths: list, img2_paths: list) -> np.ndarray:
        """
        批量预测多组图像对（前提是 ONNX 模型支持动态 batch）
        返回形状为 (N,) 的相似概率数组
        """
        imgs1 = np.vstack([preprocess_image(p, self.input_size) for p in img1_paths])
        imgs2 = np.vstack([preprocess_image(p, self.input_size) for p in img2_paths])

        ort_inputs = {
            self.input_names[0]: imgs1,
            self.input_names[1]: imgs2
        }
        logits = self.session.run(self.output_names, ort_inputs)[0]  # (N, 1)
        probs = 1.0 / (1.0 + np.exp(-logits))
        return probs.flatten()

    def reason_all_batch(self, image_1_list: list, image_2_list: list)  -> list:
        """
        批量计算两组图片之间的所有组合相似度
        :param image_1_list: 图片路径列表（或已预处理数组），长度 N
        :param image_2_list: 图片路径列表（或已预处理数组），长度 M
        :return: 二维列表 scores[N][M]，scores[i][j] 为 image_1[i] 与 image_2[j] 的相似概率
        """
        N = len(image_1_list)
        M = len(image_2_list)
        processed_1 = [preprocess_image(img) for img in image_1_list]
        processed_2 = [preprocess_image(img) for img in image_2_list]
        x1_list = []
        x2_list = []
        for p1 in processed_1:
            x1_list.extend([p1] * M)
            x2_list.extend(processed_2)
        x1_batch = np.concatenate(x1_list, axis=0)
        x2_batch = np.concatenate(x2_list, axis=0)
        ort_inputs = {self.input_names[0]: x1_batch, self.input_names[1]: x2_batch}
        logits = self.session.run(self.output_names, ort_inputs)[0]
        probs = 1.0 / (1.0 + np.exp(-logits))
        probs = probs.flatten().tolist()
        scores = [probs[i * M: (i + 1) * M] for i in range(N)]
        return scores


