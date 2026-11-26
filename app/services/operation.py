# !/usr/bin/env python
# -*-coding:utf-8 -*-

"""业务编排：请求 -> 图像字节 -> src 推理 -> 响应。

`app/` 里只有这里接触推理：不 import onnxruntime / cv2，模型细节全留在 `src/`；
阻塞操作（下载、推理、画图）都丢线程池，接口保持 async 不卡事件循环。
"""
from __future__ import annotations

import logging
import threading
import uuid
from typing import Any, Optional

from starlette.concurrency import run_in_threadpool

from src.config import load_config, section
from src.draw import annotate
from src.imageio import decode_base64, encode_jpeg, fetch_url, load_rgb
from src.pipeline import PlanResult, V8Pipeline

logger = logging.getLogger("Text_select_captcha.operation")

SCHEME_NAME = "geetest_click"
SCHEME_TITLE = "极验文字点选（v8）"
SCHEME_DESC = "YOLO 检测提示字/场景字 + 256-d embedding 余弦匹配 + 最优分配"

_pipeline: Optional[V8Pipeline] = None
_last_error: Optional[str] = None
_config_name = "v8"
_server_cfg: dict[str, Any] = {}
_lock = threading.Lock()


# --------------------------------------------------------------------------- #
def configure(config: str = "v8") -> None:
    """指定配置名并缓存 `[server]` 段；应在加载模型之前调用一次。"""
    global _config_name, _server_cfg
    _config_name = config
    _server_cfg = section(load_config(config), "server")


def max_image_bytes() -> int:
    return int(float(_server_cfg.get("max_image_mb", 5)) * 1024 * 1024)


def load_pipeline() -> V8Pipeline:
    """加载（或复用）流水线，线程安全；失败记录原因并抛出。"""
    global _pipeline, _last_error
    if _pipeline is not None:
        return _pipeline
    with _lock:
        if _pipeline is None:
            try:
                # 线程数交给配置决定（"auto" 按核数取），服务层不硬编码
                _pipeline = V8Pipeline(config=_config_name)
                _last_error = None
                logger.info(
                    "v8 模型加载完成: embed_dim=%s threads=%s/%s",
                    _pipeline.encoder.embed_dim,
                    _pipeline.detector.threads,
                    _pipeline.encoder.threads,
                )
            except Exception as exc:
                _last_error = str(exc)
                logger.warning("v8 模型加载失败: %s", exc)
                raise
    return _pipeline


# --------------------------------------------------------------------------- #
def check_size(raw: bytes) -> bytes:
    """校验图片大小，超限抛 ValueError。"""
    limit = max_image_bytes()
    if len(raw) > limit:
        raise ValueError(f"图片超过大小限制: {len(raw)} > {limit}")
    return raw


def load_image_bytes(data_type: int, image_source: str) -> bytes:
    """URL / base64 -> 图片字节。"""
    raw = fetch_url(image_source) if int(data_type) == 1 else decode_base64(image_source)
    return check_size(raw)


async def set_imageSource(dataType: int, imageSource: str) -> bytes:
    """保留原函数名与形态，内部走线程池。"""
    return await run_in_threadpool(load_image_bytes, dataType, imageSource)


def predict_bytes(raw: bytes) -> PlanResult:
    return load_pipeline().predict(raw)


def render_jpeg(raw: bytes) -> tuple[bytes, Optional[str]]:
    """识别 + 画序号，返回 (JPEG 字节, 识别错误信息)。"""
    rgb = load_rgb(raw)
    plan = load_pipeline().predict(rgb)
    if plan.boxes:
        labels = [str(i + 1) for i in range(len(plan.boxes))]
        rgb = annotate(rgb, plan.boxes, labels, plan.points)
    logger.info("标注图渲染 boxes=%d error=%s", len(plan.boxes), plan.error)
    return encode_jpeg(rgb), plan.error


# --------------------------------------------------------------------------- #
def _legacy_res(plan: PlanResult) -> dict[str, Any]:
    """v7 字段名（imgW/imgH/point/corp）。

    兼容是**服务层**的承诺：推理层不再保留 v7 词汇，只有 HTTP 响应还带这层壳。
    """
    return {
        "imgW": plan.img_w,
        "imgH": plan.img_h,
        "point": [
            {"x_rel": (b[0] + b[2]) / 2.0, "y_rel": (b[1] + b[3]) / 2.0} for b in plan.boxes
        ],
        "corp": [{"x1": b[0], "y1": b[1], "x2": b[2], "y2": b[3]} for b in plan.boxes],
    }


async def identify(item, image_id: str = "") -> dict[str, Any]:
    """识别，一次请求同时给两套字段（同一份推理结果）：

        res     {imgW, imgH, point, corp}          老调用方零改动
        boxes / points / scores / order / margin   新字段，按点击顺序

    `image_id`：调用方传了就原样回，没传就用本次请求号，再没有才随机生成。
    """
    image_id = image_id or item.imageID or uuid.uuid4().hex[:12]
    raw = await set_imageSource(item.dataType, item.imageSource)
    plan = await run_in_threadpool(predict_bytes, raw)
    logger.info(
        "识别完成 imageID=%s n_prompts=%s n_targets=%s margin=%.3f total=%.1fms error=%s",
        image_id,
        plan.n_prompts,
        plan.n_targets,
        plan.margin,
        plan.timings_ms.get("total", 0.0),
        plan.error,
    )
    return {
        "imageID": image_id,
        "scheme": SCHEME_NAME,
        "res": _legacy_res(plan),
        **plan.to_dict(),
    }


async def show(item) -> tuple[bytes, Optional[str]]:
    raw = await set_imageSource(item.dataType, item.imageSource)
    return await run_in_threadpool(render_jpeg, raw)


async def show_upload(raw: bytes) -> tuple[bytes, Optional[str]]:
    """上传场景：已经是图片字节，只做大小校验 + 渲染。"""
    return await run_in_threadpool(render_jpeg, check_size(raw))


# --------------------------------------------------------------------------- #
def health() -> dict[str, Any]:
    """懒加载前 loaded 为空、没有 spec，但 status 仍是 ok。"""
    pipe, err = _pipeline, _last_error
    body: dict[str, Any] = {
        "status": "ok" if pipe is not None or not err else "degraded",
        "loaded": [SCHEME_NAME] if pipe is not None else [],
        "errors": {SCHEME_NAME: err} if err else {},
        "scheme": SCHEME_NAME,
    }
    if pipe is not None:
        body["spec"] = pipe.spec()
    return body


def schemes() -> list[dict[str, Any]]:
    return [
        {
            "name": SCHEME_NAME,
            "title": SCHEME_TITLE,
            "description": SCHEME_DESC,
            "loaded": _pipeline is not None,
            "ready": _last_error is None,
            "missing": [],
        }
    ]
