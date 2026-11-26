# !/usr/bin/env python
# -*-coding:utf-8 -*-

"""YOLO ONNX 检测器 + letterbox 坐标还原。

- end2end: [1, 300, 6]  -> x1,y1,x2,y2,score,cls
- yolov5 : [1, N, 5+nc]-> cx,cy,w,h,obj,cls...
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import cv2
import numpy as np
import onnxruntime as ort

from .config import resolve, resolve_threads


@dataclass(frozen=True)
class Detection:
    """一个检测框（原图坐标系）。cls: 0=target 场景字 / 1=title 提示条 / 2=char 提示字。"""

    box: tuple[int, int, int, int]
    score: float
    cls: int
    label: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "box": list(self.box),
            "score": round(float(self.score), 4),
            "cls": self.cls,
            "label": self.label,
        }


def letterbox(
    rgb: np.ndarray, size: int, pad: int = 114, interp: str = "linear"
) -> tuple[np.ndarray, float, tuple[int, int]]:
    """保持宽高比缩放到 size×size，居中填充。返回 (图, 缩放比, (dw, dh))。"""
    h, w = rgb.shape[:2]
    size = int(size)
    scale = min(size / w, size / h)
    nw, nh = max(1, int(round(w * scale))), max(1, int(round(h * scale)))
    method = cv2.INTER_CUBIC if interp == "cubic" else cv2.INTER_LINEAR
    resized = cv2.resize(rgb, (nw, nh), interpolation=method)
    canvas = np.full((size, size, 3), int(pad), dtype=np.uint8)
    dw, dh = (size - nw) // 2, (size - nh) // 2
    canvas[dh:dh + nh, dw:dw + nw] = resized
    return canvas, scale, (dw, dh)


def unletterbox(
    boxes: np.ndarray, scale: float, pad: tuple[int, int], orig_w: int, orig_h: int
) -> np.ndarray:
    """[N,4] xyxy 从填充坐标系换算回原图，并裁剪到图内。"""
    out = boxes.astype(np.float32).copy()
    if out.size == 0:
        return out
    out[:, [0, 2]] = np.clip((out[:, [0, 2]] - pad[0]) / max(scale, 1e-6), 0, orig_w)
    out[:, [1, 3]] = np.clip((out[:, [1, 3]] - pad[1]) / max(scale, 1e-6), 0, orig_h)
    return out


def _nms(boxes: np.ndarray, scores: np.ndarray, iou_thr: float) -> list[int]:
    """单类 NMS。"""
    if len(boxes) == 0:
        return []
    x1, y1, x2, y2 = boxes[:, 0], boxes[:, 1], boxes[:, 2], boxes[:, 3]
    areas = np.maximum(0, x2 - x1) * np.maximum(0, y2 - y1)
    order = np.argsort(-scores)
    keep: list[int] = []
    while order.size > 0:
        i = int(order[0])
        keep.append(i)
        if order.size == 1:
            break
        rest = order[1:]
        inter = np.maximum(0, np.minimum(x2[i], x2[rest]) - np.maximum(x1[i], x1[rest])) * np.maximum(
            0, np.minimum(y2[i], y2[rest]) - np.maximum(y1[i], y1[rest])
        )
        order = rest[inter / np.maximum(areas[i] + areas[rest] - inter, 1e-6) <= iou_thr]
    return keep


class YoloDetector:
    """ONNX YOLO 检测器；classes 顺序决定 cls 含义。"""

    def __init__(self, cfg: dict):
        self.path = resolve(cfg["model"])
        self.classes = list(cfg.get("classes", []))
        self.conf = float(cfg.get("conf", 0.3))
        self.iou = float(cfg.get("iou", 0.5))
        self.pad = int(cfg.get("pad", 114))
        self.interp = str(cfg.get("interp", "linear"))
        self.fmt = str(cfg.get("format", "auto"))

        so = ort.SessionOptions()
        self.threads = resolve_threads(cfg.get("threads"))
        so.intra_op_num_threads = self.threads
        so.add_session_config_entry("session.intra_op.allow_spinning", "0")
        so.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        self.session = ort.InferenceSession(str(self.path), so, providers=["CPUExecutionProvider"])
        self.input_name = self.session.get_inputs()[0].name
        shape = self.session.get_inputs()[0].shape
        self.input_size = int(
            cfg.get("input_size")
            or (shape[2] if len(shape) > 2 and isinstance(shape[2], int) else 640)
        )

    def _format(self, out: np.ndarray) -> str:
        if self.fmt != "auto":
            return self.fmt
        return "end2end" if out.ndim == 3 and out.shape[-1] == 6 and out.shape[1] <= 1000 else "yolov5"

    def detect(self, rgb: np.ndarray, conf: float | None = None) -> list[Detection]:
        """rgb: HWC uint8 RGB。返回原图坐标系下的检测框。"""
        h, w = rgb.shape[:2]
        canvas, scale, pad = letterbox(rgb, self.input_size, self.pad, self.interp)
        x = np.transpose(canvas.astype(np.float32) / 255.0, (2, 0, 1))[None]
        out = self.session.run(None, {self.input_name: x})[0]
        threshold = self.conf if conf is None else float(conf)

        if self._format(out) == "end2end":
            rows = out[0]
            boxes, scores, classes = rows[:, :4], rows[:, 4], rows[:, 5].astype(np.int64)
            mask = scores >= threshold
            boxes, scores, classes = boxes[mask], scores[mask], classes[mask]
        else:
            pred = out[0]
            cls_scores = pred[:, 5:]
            cls_id = cls_scores.argmax(axis=1)
            scores = pred[:, 4] * cls_scores[np.arange(len(pred)), cls_id]
            mask = scores >= threshold
            cxcywh, scores, classes = pred[mask, :4], scores[mask], cls_id[mask]

            cx, cy, bw, bh = cxcywh[:, 0], cxcywh[:, 1], cxcywh[:, 2], cxcywh[:, 3]
            boxes = np.stack([cx - bw / 2, cy - bh / 2, cx + bw / 2, cy + bh / 2], axis=1)

            keep: list[int] = []
            for c in np.unique(classes):
                idx = np.where(classes == c)[0]
                keep.extend(idx[_nms(boxes[idx], scores[idx], self.iou)].tolist())
            k = np.array(sorted(keep), dtype=np.int64)
            boxes, scores, classes = boxes[k], scores[k], classes[k]

        if len(boxes) == 0:
            return []

        dets: list[Detection] = []
        for box, score, cls in zip(unletterbox(boxes, scale, pad, w, h), scores, classes):
            x1, y1, x2, y2 = [int(v) for v in box]
            if x2 - x1 < 4 or y2 - y1 < 4:
                continue
            cls = int(cls)
            label = self.classes[cls] if 0 <= cls < len(self.classes) else str(cls)
            dets.append(Detection((x1, y1, x2, y2), float(score), cls, label))
        return dets
