# !/usr/bin/env python
# -*-coding:utf-8 -*-

"""
# File       : draw.py
# Time       ：2026/11/3 11:20
# Description：v8 结果可视化（框 + 点击序号 + 中心十字）。
"""
from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np

from .imageio import save_rgb

_COLORS = [
    (0, 0, 255), (0, 200, 0), (255, 0, 0), (0, 180, 255),
    (255, 0, 255), (255, 255, 0), (128, 0, 255), (0, 128, 128),
]


def annotate(
    rgb: np.ndarray,
    boxes,
    labels: list[str] | None = None,
    points=None,
    thickness: int = 2,
) -> np.ndarray:
    """在 RGB 图上画框与序号，返回新的 RGB 图。"""
    img = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR).copy()
    for i, box in enumerate(boxes):
        x1, y1, x2, y2 = [int(v) for v in box]
        color = _COLORS[i % len(_COLORS)]
        cv2.rectangle(img, (x1, y1), (x2, y2), color, thickness=thickness, lineType=cv2.LINE_AA)

        label = labels[i] if labels and i < len(labels) else str(i + 1)
        (tw, th), _ = cv2.getTextSize(str(label), cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
        cv2.rectangle(img, (x1, max(0, y1 - th - 6)), (x1 + tw + 6, y1), color, -1)
        cv2.putText(
            img, str(label), (x1 + 3, max(12, y1 - 4)),
            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2,
        )
        if points is not None and i < len(points):
            cv2.drawMarker(
                img, (int(points[i][0]), int(points[i][1])), color,
                markerType=cv2.MARKER_CROSS, markerSize=10, thickness=1,
            )
    return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)


def save_annotated(
    rgb: np.ndarray,
    boxes,
    path: str | Path,
    labels: list[str] | None = None,
    points=None,
) -> Path:
    return save_rgb(annotate(rgb, boxes, labels, points), path)
