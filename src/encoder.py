# !/usr/bin/env python
# -*-coding:utf-8 -*-

"""v8 encoder：预处理 + MobileNetV4-conv-small + 多尺度 GeM。

预处理**必须与训练侧逐位一致**，否则同一张裁剪图会编码到别的向量上：

    RGB -> letterbox(112, pad=128) -> /255
        -> RGB 逐图逐通道标准化 -> Sobel 第 4 通道 -> CHW float32

输入 [B,4,112,112] -> 输出 [B,256]（已 L2 归一化）。
同字不同渲染风格向量接近、不同字远离，所以直接对 N×M 余弦矩阵做最优分配。
"""
from __future__ import annotations

from typing import Sequence

import cv2
import numpy as np
import onnxruntime as ort

from .config import resolve, resolve_threads


# --------------------------------------------------------------------------- #
# 预处理
# --------------------------------------------------------------------------- #
def letterbox(rgb: np.ndarray, size: int, pad_value: int = 128) -> np.ndarray:
    """按短边等比缩放到 size×size 并居中填充（缩小用 AREA，放大用 CUBIC）。"""
    h = w = int(size)
    ih, iw = rgb.shape[:2]
    scale = min(w / iw, h / ih)
    nw, nh = max(1, int(round(iw * scale))), max(1, int(round(ih * scale)))
    resized = cv2.resize(
        rgb, (nw, nh), interpolation=cv2.INTER_AREA if scale < 1.0 else cv2.INTER_CUBIC
    )
    canvas = np.full((h, w, 3), int(pad_value), dtype=np.uint8)
    dx, dy = (w - nw) // 2, (h - nh) // 2
    canvas[dy:dy + nh, dx:dx + nw] = resized
    return canvas


def sobel_channel(rgb: np.ndarray, mode: str = "fixed") -> np.ndarray:
    """Sobel 梯度幅值（第 4 通道），削弱「颜色/背景」这条捷径。

    mode='fixed'（训练侧默认）= mag/4；mode='per_image' = 按本图最大值归一化。
    """
    gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY).astype(np.float32) / 255.0
    gx = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3)
    mag = cv2.magnitude(gx, gy)
    mag = mag / (float(mag.max()) + 1e-6) if mode == "per_image" else mag / 4.0
    return np.clip(mag, 0.0, 1.0).astype(np.float32)


def to_tensor(
    rgb: np.ndarray,
    input_size: int = 112,
    pad_value: int = 128,
    channels: int = 4,
    standardize: str = "per_image_channel",
    sobel_norm: str = "fixed",
) -> np.ndarray:
    """单张 RGB 裁剪图 -> [C,H,W] float32。"""
    lb = letterbox(rgb, input_size, pad_value)
    x = lb.astype(np.float32) / 255.0

    if standardize == "per_image_channel":
        x = (x - x.mean(axis=(0, 1), keepdims=True)) / np.maximum(
            x.std(axis=(0, 1), keepdims=True), 1e-6
        )
    elif standardize == "per_image_global":
        x = (x - float(x.mean())) / max(float(x.std()), 1e-6)
    elif standardize != "none":
        raise ValueError(f"未知 standardize: {standardize}")

    if int(channels) == 4:
        x = np.concatenate([x, sobel_channel(lb, sobel_norm)[:, :, None]], axis=2)
    elif int(channels) != 3:
        raise ValueError(f"channels 只支持 3/4，收到 {channels}")

    return np.ascontiguousarray(np.transpose(x, (2, 0, 1))).astype(np.float32)


# --------------------------------------------------------------------------- #
# 推理
# --------------------------------------------------------------------------- #
def _session_options(threads) -> ort.SessionOptions:
    so = ort.SessionOptions()
    so.intra_op_num_threads = resolve_threads(threads)
    # 线上单请求场景关闭自旋，避免多进程互相抢核
    so.add_session_config_entry("session.intra_op.allow_spinning", "0")
    so.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    return so


class V8Encoder:
    """把一个裁剪图编码成 256-d 向量。"""

    def __init__(self, model_path: str, max_batch: int = 32, threads=None):
        """threads 只由 V8Pipeline 从配置注入（"auto" = 按 CPU 核数），不对外暴露。"""
        self.model_path = str(resolve(model_path))
        self.threads = resolve_threads(threads)
        self.session = ort.InferenceSession(
            self.model_path, _session_options(self.threads), providers=["CPUExecutionProvider"]
        )
        self.input_name = self.session.get_inputs()[0].name
        self.output_name = self.session.get_outputs()[0].name
        self.max_batch = max(1, int(max_batch))
        shape = self.session.get_outputs()[0].shape
        self.embed_dim = int(shape[1]) if len(shape) > 1 and isinstance(shape[1], int) else 0

    def embed_tensors(self, tensors: Sequence[np.ndarray]) -> np.ndarray:
        """[N,C,H,W] 列表 -> [N,embed_dim]，按 max_batch 分块。"""
        if not tensors:
            return np.zeros((0, self.embed_dim or 0), dtype=np.float32)
        outputs = []
        for start in range(0, len(tensors), self.max_batch):
            chunk = np.stack(tensors[start:start + self.max_batch]).astype(np.float32)
            outputs.append(self.session.run([self.output_name], {self.input_name: chunk})[0])
        return np.concatenate(outputs, axis=0).astype(np.float32)

    def tensors_from_rgbs(self, rgbs: Sequence[np.ndarray], prep: dict) -> list[np.ndarray]:
        return [
            to_tensor(
                rgb,
                input_size=prep["input_size"],
                pad_value=prep["pad_value"],
                channels=prep["channels"],
                standardize=prep["standardize"],
                sobel_norm=prep["sobel_norm"],
            )
            for rgb in rgbs
        ]

    def embed_rgbs(self, rgbs: Sequence[np.ndarray], prep: dict) -> np.ndarray:
        return self.embed_tensors(self.tensors_from_rgbs(rgbs, prep))
