# !/usr/bin/env python
# -*-coding:utf-8 -*-

"""HTTP 访问日志中间件：记录入参、出参、业务码与耗时。

两条约束：
1. **图片绝不进日志**：请求里的 base64 只记长度，上传文件只记文件名+大小，
   响应是 image/* 时只记 content-type 和字节数。
2. 单个参数超过 `MAX_STR` 字符就替换成 `<省略 N 字节>`，一行日志再截断到 `MAX_BODY`。
"""
from __future__ import annotations

import json
import logging
import time
import uuid
from typing import Any, Callable

logger = logging.getLogger("Text_select_captcha.access")

MAX_STR = 200      # 单个字符串参数的记录上限（base64 动辄几十万字符）
MAX_BODY = 4000    # 一行日志里 JSON 的最大长度


def _shrink(value: Any) -> Any:
    """超长字符串替换成占位，避免日志被图片数据撑爆。"""
    if isinstance(value, dict):
        return {k: _shrink(v) for k, v in value.items()}
    if isinstance(value, list):
        return [_shrink(v) for v in value]
    if isinstance(value, str) and len(value) > MAX_STR:
        return f"<省略 {len(value)} 字节>"
    return value


def _dump(value: Any) -> str:
    text = json.dumps(value, ensure_ascii=False)
    return text if len(text) <= MAX_BODY else f"{text[:MAX_BODY]}...<共 {len(text)} 字节>"


def _request_params(content_type: str, body: bytes) -> str:
    if not body:
        return "-"
    if content_type.startswith("application/json"):
        try:
            return _dump(_shrink(json.loads(body)))
        except ValueError:
            return f"<json 解析失败 {len(body)}B>"
    if content_type.startswith("multipart/form-data"):
        # 上传文件只留文件名，内容一个字节都不记
        marker = b'filename="'
        start = body.find(marker)
        name = "?"
        if start >= 0:
            start += len(marker)
            end = body.find(b'"', start)
            if end > start:
                name = body[start:end].decode("utf-8", "replace")
        return f"<文件 {name} {len(body)}B>"
    return f"<{len(body)}B>"


def _response_summary(content_type: str, buffered: bytes, total: int) -> tuple[Any, str]:
    """返回 (业务码, 摘要)；图片只记大小。"""
    if content_type.startswith("application/json") and buffered:
        try:
            data = json.loads(buffered)
        except ValueError:
            return None, f"<json 解析失败 {total}B>"
        if "code" in data or "data" in data:  # 标准的 code/msg/data 信封
            return data.get("code"), _dump(_shrink(data.get("data")))
        return None, _dump(_shrink(data))     # /health 这类裸结构，整体记下来
    if content_type.startswith("image/"):
        return None, f"<{content_type} {total}B>"
    return None, f"<{total}B>"


class AccessLogMiddleware:
    """纯 ASGI 中间件：读完 body 再回放给下游，同时收集响应做日志。"""

    def __init__(self, app: Callable):
        self.app = app

    async def __call__(self, scope, receive, send):
        if scope["type"] != "http":
            await self.app(scope, receive, send)
            return

        rid = uuid.uuid4().hex[:8]
        started = time.perf_counter()
        # 让接口能拿到本次请求号（业务没传 imageID 时直接拿它当 imageID，日志和响应就对上了）
        scope.setdefault("state", {})["request_id"] = rid

        # 先读完请求体，再原样回放 —— BaseHTTPMiddleware 直接读 body 会让下游拿到空流
        body = b""
        while True:
            message = await receive()
            if message["type"] == "http.disconnect":
                break
            body += message.get("body", b"")
            if not message.get("more_body"):
                break

        replayed = False

        async def replay():
            nonlocal replayed
            if replayed:
                return {"type": "http.disconnect"}
            replayed = True
            return {"type": "http.request", "body": body, "more_body": False}

        status = 0
        headers: dict[str, str] = {}
        total = 0
        buffered = b""

        async def send_logging(message):
            nonlocal status, headers, total, buffered
            if message["type"] == "http.response.start":
                status = message["status"]
                raw = message.get("headers", [])
                headers = {k.decode().lower(): v.decode("latin-1") for k, v in raw}
                message = {**message, "headers": [*raw, (b"x-request-id", rid.encode())]}
            elif message["type"] == "http.response.body":
                chunk = message.get("body", b"")
                total += len(chunk)
                if len(buffered) <= MAX_BODY and headers.get("content-type", "").startswith("application/json"):
                    buffered += chunk
            await send(message)

        method = scope.get("method", "-")
        query = scope.get("query_string", b"").decode("latin-1")
        path = scope["path"] + (f"?{query}" if query else "")
        client = (scope.get("client") or ("-", 0))[0]
        content_type = {k.decode().lower(): v.decode("latin-1") for k, v in scope.get("headers", [])}.get(
            "content-type", ""
        )

        logger.info(
            "→ %s %s from=%s id=%s ct=%s len=%d params=%s",
            method, path, client, rid, content_type or "-", len(body),
            _request_params(content_type, body),
        )

        try:
            await self.app(scope, replay, send_logging)
        except Exception:
            logger.exception("✗ %s %s id=%s", method, path, rid)
            raise
        finally:
            elapsed = (time.perf_counter() - started) * 1000
            code, summary = _response_summary(headers.get("content-type", ""), buffered, total)
            logger.info(
                "← %s %s %s%s %.1fms out=%s",
                method,
                path,
                status,
                f" code={code}" if code is not None else "",
                elapsed,
                summary,
            )
