# !/usr/bin/env python
# -*-coding:utf-8 -*-

"""FastAPI 应用工厂。

    python service.py                     # 推荐入口
    uvicorn app.main:app                  # 模块级 app
    uvicorn app.main:create_app --factory # 工厂模式（默认懒加载）
"""
from __future__ import annotations

import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse

from app.api.router import router as v1_router
from app.middleware import AccessLogMiddleware
from app.services import operation
from app.utils.errors import ApiError, bad_error, ok
from app.utils.log import setup_logging
from src.config import load_config, section

logger = logging.getLogger("Text_select_captcha.app")


def create_app(eager: bool = False, config: str = "v8") -> FastAPI:
    """eager=True 启动即加载模型；失败不影响启动，原因进 /health。"""
    cfg = load_config(config)
    setup_logging(section(cfg, "log"))
    operation.configure(config)

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        if eager:
            try:
                operation.load_pipeline()
            except Exception as exc:  # 缺模型也能起服务，/health 里能看到原因
                logger.warning("v8 模型预加载失败: %s", exc)
        yield

    app = FastAPI(title="文字点选验证码识别服务（v8）", version="2.1.0", lifespan=lifespan)
    app.add_middleware(AccessLogMiddleware)

    # 统一错误：HTTP 恒 200，业务码在 body
    @app.exception_handler(ApiError)
    async def api_error(request: Request, exc: ApiError):
        return JSONResponse(status_code=200, content=bad_error(exc.detail, exc.code))

    @app.exception_handler(RequestValidationError)
    async def invalid_request(request: Request, exc: RequestValidationError):
        return JSONResponse(status_code=200, content=bad_error("请求参数错误", 400))

    @app.get("/", summary="存活探针")
    def root():
        return ok("ok")

    @app.get("/health", summary="健康检查（含模型规格）")
    def health():
        return operation.health()

    app.include_router(v1_router, prefix="/api/v1")
    return app


app = create_app()  # 供 uvicorn app.main:app 直接导入
