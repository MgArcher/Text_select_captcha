# !/usr/bin/env python
# -*-coding:utf-8 -*-

"""检测 + 识别流水线：检测 -> 裁剪 -> embedding -> 最优分配。

老方案要 N×M 次 Siamese 前向；v8 只做 N+M 次 encoder 前向 + 一次点积。
对外接口：`predict` / `detection` / `spec`（v7 的 run/run_dict 已移除）。
"""
from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any

import numpy as np

from .assign import exact_assignment
from .config import load_config, section
from .detector import Detection, YoloDetector
from .encoder import V8Encoder
from .imageio import load_rgb


@dataclass
class PlanResult:
    """一次识别结果：按点击顺序排列的目标框 + 点 + 得分。"""

    boxes: list[list[int]] = field(default_factory=list)
    points: list[list[float]] = field(default_factory=list)
    scores: list[float] = field(default_factory=list)
    order: list[int] = field(default_factory=list)
    margin: float = 0.0
    n_prompts: int = 0
    n_targets: int = 0
    img_w: int = 0
    img_h: int = 0
    timings_ms: dict[str, float] = field(default_factory=dict)
    error: str | None = None
    warning: str | None = None

    @property
    def ok(self) -> bool:
        return self.error is None and bool(self.boxes)

    def to_dict(self) -> dict[str, Any]:
        data: dict[str, Any] = {
            "boxes": self.boxes,
            "points": self.points,
            "scores": self.scores,
            "order": self.order,
            "margin": round(float(self.margin), 4),
            "n_prompts": self.n_prompts,
            "n_targets": self.n_targets,
            "timings_ms": {k: round(float(v), 2) for k, v in self.timings_ms.items()},
            "error": self.error,
        }
        if self.warning:
            data["warning"] = self.warning
        return data


class V8Pipeline:
    """极验文字点选：v8 encoder + YOLO 检测器。"""

    name = "geetest_click"
    title = "极验文字点选（v8）"

    def __init__(self, config: str = "v8"):
        """线程数由配置决定（`threads = "auto"` 按 CPU 核数），不开放入参。"""
        cfg = load_config(config)
        det_cfg = section(cfg, "detector")
        enc_cfg = section(cfg, "encoder")
        sch_cfg = section(cfg, "scheme")

        self.detector = YoloDetector(det_cfg)
        self.encoder = V8Encoder(
            enc_cfg["model"],
            max_batch=int(enc_cfg.get("max_batch", 32)),
            threads=enc_cfg.get("threads"),
        )
        self.prep = {
            "input_size": int(enc_cfg.get("input_size", 112)),
            "pad_value": int(enc_cfg.get("pad_value", 128)),
            "channels": int(enc_cfg.get("channels", 4)),
            "standardize": str(enc_cfg.get("standardize", "per_image_channel")),
            "sobel_norm": str(enc_cfg.get("sobel_norm", "fixed")),
        }
        self.target_cls = int(sch_cfg.get("target_cls", 0))
        self.prompt_cls = int(sch_cfg.get("prompt_cls", 2))
        self.title_cls = int(sch_cfg.get("title_cls", 1))
        self.crop_pad = int(sch_cfg.get("crop_pad", 0))
        self.use_title_filter = bool(sch_cfg.get("use_title_filter", True))
        self.adaptive_conf = bool(sch_cfg.get("adaptive_conf", True))
        self.fallback_conf = float(sch_cfg.get("fallback_conf", 0.15))

    # ------------------------------------------------------------------ #
    def _split(self, dets: list[Detection]) -> tuple[list[Detection], list[Detection]]:
        """分成 (场景字, 提示字)；提示字按 x 从左到右，并用提示条位置过滤误检。"""
        targets = [d for d in dets if d.cls == self.target_cls]
        prompts = sorted(
            (d for d in dets if d.cls == self.prompt_cls),
            key=lambda d: (d.box[0] + d.box[2]) / 2.0,
        )
        if not self.use_title_filter or not prompts:
            return targets, prompts

        titles = [d for d in dets if d.cls == self.title_cls]
        kept = [
            d
            for d in prompts
            if any(
                t.box[0] - 0.2 * (t.box[2] - t.box[0]) <= (d.box[0] + d.box[2]) / 2.0
                <= t.box[2] + 0.2 * (t.box[2] - t.box[0])
                and t.box[1] - 0.2 * (t.box[3] - t.box[1]) <= (d.box[1] + d.box[3]) / 2.0
                <= t.box[3] + 0.2 * (t.box[3] - t.box[1])
                for t in titles
            )
        ]
        return targets, kept or prompts

    @staticmethod
    def _crop(rgb: np.ndarray, box: tuple[int, int, int, int], pad: int) -> np.ndarray:
        """按框裁剪（可外扩 pad 像素），并夹到图内。"""
        h, w = rgb.shape[:2]
        x1, y1, x2, y2 = box
        x1, y1 = max(0, min(w - 1, x1 - pad)), max(0, min(h - 1, y1 - pad))
        x2, y2 = max(x1 + 1, min(w, x2 + pad)), max(y1 + 1, min(h, y2 + pad))
        return rgb[y1:y2, x1:x2]

    # ------------------------------------------------------------------ #
    def detection(self, image) -> list[list[float]]:
        """只做检测：[[x1,y1,x2,y2,score,cls], ...]。"""
        return [
            [int(d.box[0]), int(d.box[1]), int(d.box[2]), int(d.box[3]), float(d.score), int(d.cls)]
            for d in self.detector.detect(load_rgb(image))
        ]

    def predict(self, image, **_: Any) -> PlanResult:
        """完整流水线。image 支持 路径 / bytes / base64 bytes / ndarray / PIL。"""
        t0 = time.perf_counter()
        rgb = load_rgb(image)
        h, w = rgb.shape[:2]
        t1 = time.perf_counter()

        dets = self.detector.detect(rgb)
        targets, prompts = self._split(dets)

        # 提示字漏检（阈值偏高）时用低阈值补检，数量对上了才采用
        if self.adaptive_conf and len(targets) >= 2 and len(prompts) < len(targets):
            dets_low = self.detector.detect(rgb, conf=self.fallback_conf)
            targets_low, prompts_low = self._split(dets_low)
            if len(targets_low) == len(targets) == len(prompts_low) >= 2:
                targets, prompts = targets_low, prompts_low
        t2 = time.perf_counter()

        result = PlanResult(
            n_prompts=len(prompts),
            n_targets=len(targets),
            img_w=w,
            img_h=h,
            timings_ms={"decode": (t1 - t0) * 1000, "detect": (t2 - t1) * 1000},
        )
        if not prompts or not targets:
            result.error = f"检测不足: prompts={len(prompts)}, targets={len(targets)}"
            result.timings_ms["total"] = (time.perf_counter() - t0) * 1000
            return result

        prompt_tensors = self.encoder.tensors_from_rgbs(
            [self._crop(rgb, d.box, self.crop_pad) for d in prompts], self.prep
        )
        target_tensors = self.encoder.tensors_from_rgbs(
            [self._crop(rgb, d.box, self.crop_pad) for d in targets], self.prep
        )
        t3 = time.perf_counter()

        prompt_emb = self.encoder.embed_tensors(prompt_tensors)
        target_emb = self.encoder.embed_tensors(target_tensors)
        t4 = time.perf_counter()

        sim = prompt_emb @ target_emb.T  # 两边都是 L2 归一化，点积即余弦
        perm, _, margin = exact_assignment(sim)
        t5 = time.perf_counter()

        for i, j in enumerate(perm):
            if j < 0:
                continue
            x1, y1, x2, y2 = targets[j].box
            result.boxes.append([x1, y1, x2, y2])
            result.points.append([(x1 + x2) / 2.0, (y1 + y2) / 2.0])
            result.scores.append(round(float(sim[i, j]), 4))
            result.order.append(i)
        result.margin = float(margin)
        if len(result.boxes) < len(prompts):
            result.warning = f"部分提示字未匹配: {len(prompts) - len(result.boxes)}"

        result.timings_ms.update(
            {
                "preprocess": (t3 - t2) * 1000,
                "encode": (t4 - t3) * 1000,
                "assign": (t5 - t4) * 1000,
                "total": (t5 - t0) * 1000,
            }
        )
        return result

    def spec(self) -> dict[str, Any]:
        """健康检查用：模型与预处理信息。"""
        return {
            "scheme": self.name,
            "detector": str(self.detector.path),
            "encoder": self.encoder.model_path,
            "embed_dim": self.encoder.embed_dim,
            "input_size": self.prep["input_size"],
            "channels": self.prep["channels"],
            "classes": list(self.detector.classes),
            "crop_pad": self.crop_pad,
            "adaptive_conf": self.adaptive_conf,
        }
