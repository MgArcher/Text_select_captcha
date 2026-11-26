# !/usr/bin/env python
# -*-coding:utf-8 -*-

"""文字点选检测 + 识别引擎（纯 ONNX Runtime，不依赖 PyTorch，也不依赖 Web 框架）。

本包只做推理，HTTP 服务在 `app/`：

    from src import V8Pipeline

    pipe = V8Pipeline()
    plan = pipe.predict("captcha.jpg")
    print(plan.boxes, plan.points, plan.margin)
    print(plan.to_dict())                 # 主要输出
"""
from .assign import exact_assignment, greedy_assignment
from .detector import Detection, YoloDetector
from .imageio import decode_base64, encode_jpeg, load_rgb
from .pipeline import PlanResult, V8Pipeline

__version__ = "1.0.0"

__all__ = [
    "V8Pipeline",
    "Detection",
    "PlanResult",
    "YoloDetector",
    "exact_assignment",
    "greedy_assignment",
    "load_rgb",
    "encode_jpeg",
    "decode_base64",
]
