# -*- coding: utf-8 -*-

"""业务异常 + 统一响应体。

约定：HTTP 恒 200，业务码放 body（400 参数/解码、404 方案、500 识别失败）。
所以是普通 Exception，由 app.main 统一转成 200 + 业务码。
"""
from __future__ import annotations

from typing import Any


class ApiError(Exception):
    code = 400
    default_detail = "请求参数错误"

    def __init__(self, detail: str | None = None):
        self.detail = detail or self.default_detail
        super().__init__(self.detail)


def ok(data: Any = None, msg: str = "成功") -> dict[str, Any]:
    return {"code": 200, "msg": msg, "data": data}


def bad_error(msg: str = "失败", code: int = 400) -> dict[str, Any]:
    return {"code": code, "msg": msg, "data": None}
