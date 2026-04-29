# -*- coding: utf-8 -*-
from fastapi import HTTPException

class BadRequestError(HTTPException):
    def __init__(self, detail: str = "请求参数错误"):
        super().__init__(status_code=400, detail=detail)

class ServiceError(HTTPException):
    def __init__(self, detail: str = "服务内部错误"):
        super().__init__(status_code=500, detail=detail)

def bad_error():
    """返回统一错误响应（兼容原返回格式）"""
    return {"code": 400, "msg": "失败", "data": None}