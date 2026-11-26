# !/usr/bin/env python
# -*-coding:utf-8 -*-

"""
# File       : imageio.py
# Time       ：2026/11/3 11:20
# Description：统一图像输入（路径 / URL / bytes / base64 / ndarray / PIL），统一产出 RGB。
"""
from __future__ import annotations

import base64
import urllib.request
from pathlib import Path

import cv2
import numpy as np
from PIL import Image


def _bytes_to_rgb(data: bytes) -> np.ndarray:
    """bytes -> RGB uint8；自动处理灰度图与带 alpha 的 PNG。"""
    arr = np.frombuffer(data, dtype=np.uint8)
    img = cv2.imdecode(arr, cv2.IMREAD_UNCHANGED)
    if img is None:
        raise ValueError("无法解码图像")
    if img.ndim == 2:
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    if img.shape[2] == 4:
        alpha = img[:, :, 3:4].astype(np.float32) / 255.0
        img = img[:, :, :3].astype(np.float32) * alpha + 255.0 * (1.0 - alpha)
        img = img.astype(np.uint8)
    return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)


def fetch_url(url: str, timeout: float = 10.0) -> bytes:
    req = urllib.request.Request(url, headers={"User-Agent": "Text-select-captcha-v8/1.0"})
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        return resp.read()


def decode_base64(text: str) -> bytes:
    """兼容 `data:image/jpeg;base64,xxx` 前缀。"""
    if "," in text and text.strip().lower().startswith("data:"):
        text = text.split(",", 1)[1]
    return base64.b64decode(text)


def load_rgb(source) -> np.ndarray:
    """统一返回 RGB uint8 [H,W,3]。"""
    if isinstance(source, np.ndarray):
        if source.ndim == 2:
            return cv2.cvtColor(source, cv2.COLOR_GRAY2RGB)
        if source.ndim != 3 or source.shape[2] not in (3, 4):
            raise ValueError(f"不支持的 ndarray 形状: {source.shape}")
        if source.shape[2] == 4:
            alpha = source[:, :, 3:4].astype(np.float32) / 255.0
            rgb = source[:, :, :3].astype(np.float32) * alpha + 255.0 * (1.0 - alpha)
            return rgb.astype(np.uint8)
        return np.ascontiguousarray(source)
    if isinstance(source, Image.Image):
        return load_rgb(np.asarray(source.convert("RGB")))
    if isinstance(source, (bytes, bytearray)):
        return _bytes_to_rgb(bytes(source))
    if isinstance(source, (str, Path)):
        text = str(source)
        if text.startswith(("http://", "https://")):
            return _bytes_to_rgb(fetch_url(text))
        return _bytes_to_rgb(Path(text).read_bytes())
    raise TypeError(f"不支持的图像输入类型: {type(source)}")


def encode_jpeg(rgb: np.ndarray, quality: int = 90) -> bytes:
    ok, buf = cv2.imencode(
        ".jpg", cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR), [cv2.IMWRITE_JPEG_QUALITY, quality]
    )
    if not ok:
        raise RuntimeError("JPEG 编码失败")
    return buf.tobytes()


def save_rgb(rgb: np.ndarray, path: str | Path) -> Path:
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_bytes(encode_jpeg(rgb))
    return out
