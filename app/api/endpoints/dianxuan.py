# !/usr/bin/env python
# -*-coding:utf-8 -*-

"""文字点选端点。

    POST /api/v1/identify  识别：res{imgW,imgH,point,corp} + boxes/points/... 一次给全
    POST /api/v1/show      标注 JPEG（来源 URL / base64）
    POST /api/v1/upload    标注 JPEG（来源 multipart 上传文件）

`imageID` 三个接口都支持：传了就原样回，没传就拿本次请求号（与日志的 `id=` 一致），
这样拿一张图的结果去翻日志能直接对上。
"""
from __future__ import annotations

import logging
import uuid
from typing import Optional
from urllib.parse import quote

from fastapi import APIRouter, File, Form, Request, UploadFile
from fastapi.responses import JSONResponse, Response

from app.models.input import Input
from app.services import operation
from app.utils.errors import bad_error, ok

logger = logging.getLogger("Text_select_captcha.dianxuan")

router = APIRouter()


def _resolve_id(given: str, request: Request) -> str:
    """传了原样用；没传用本次请求号；再没有才随机生成。"""
    return given or getattr(request.state, "request_id", "") or uuid.uuid4().hex[:12]


def _jpeg(img: bytes, error: Optional[str], *, image_id: str, filename: str = "") -> Response:
    headers = {"X-Code": "500" if error else "200", "X-Image-ID": image_id}
    if filename:
        headers["X-Filename"] = quote(filename, safe="")  # 头只能是 latin-1
    return Response(content=img, media_type="image/jpeg", headers=headers)


@router.post("/identify", summary="图文点选识别（新旧字段合并返回）")
async def identify(Item: Input, request: Request):
    try:
        data = await operation.identify(Item, _resolve_id(Item.imageID or "", request))
    except ValueError as exc:  # 坏 base64 / 解码失败 / 超过大小限制
        return bad_error(str(exc), 400)
    except Exception as exc:  # noqa: BLE001
        logger.exception("identify failed")
        return bad_error(str(exc), 500)

    if data.get("error"):
        return JSONResponse(status_code=200, content={"code": 500, "msg": data["error"], "data": data})
    return ok(data)


@router.post("/show", summary="识别后图片效果（URL / base64）")
async def show(Item: Input, request: Request):
    try:
        img, error = await operation.show(Item)
    except ValueError as exc:
        return bad_error(str(exc), 400)
    except Exception as exc:  # noqa: BLE001
        logger.exception("show failed")
        return bad_error(str(exc), 500)
    return _jpeg(img, error, image_id=_resolve_id(Item.imageID or "", request))


@router.post("/upload", summary="上传图片文件，返回标注后的 JPEG")
async def upload(request: Request, file: UploadFile = File(..., description="验证码图片文件"),
                 imageID: str = Form("", description="可选，原样回传")):
    """与 /show 同一条渲染路径，只是图片来源变成上传文件。"""
    try:
        raw = await file.read()
        if not raw:
            raise ValueError("上传的文件为空")
        img, error = await operation.show_upload(raw)
    except ValueError as exc:  # 空文件 / 超限 / 解码失败
        return bad_error(str(exc), 400)
    except Exception as exc:  # noqa: BLE001
        logger.exception("upload failed")
        return bad_error(str(exc), 500)
    finally:
        await file.close()
    return _jpeg(img, error, image_id=_resolve_id(imageID, request), filename=file.filename or "")
