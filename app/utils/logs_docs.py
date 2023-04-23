# !/usr/bin/env python
# -*-coding:utf-8 -*-

"""
# File       : logs_docs.py
# Time       ：2022/7/25 16:32
# Author     ：yujia
# version    ：python 3.6
# Description：
"""
import os
import logging
import uvicorn
from fastapi.openapi.docs import (
    get_redoc_html,
    get_swagger_ui_html,
    get_swagger_ui_oauth2_redirect_html,
)
from fastapi.staticfiles import StaticFiles


def get_docs(app):
    root = os.path.abspath(os.path.dirname(os.path.dirname(__file__)))
    app.mount("/static", StaticFiles(directory=f"{root}/static"), name="static")

    @app.get("/docs", include_in_schema=False)
    async def custom_swagger_ui_html():
        return get_swagger_ui_html(
            openapi_url=app.openapi_url,
            title=app.title + " - Swagger UI",
            oauth2_redirect_url=app.swagger_ui_oauth2_redirect_url,
            swagger_js_url="/static/swagger-ui-bundle.js",
            swagger_css_url="/static/swagger-ui.css",
        )

    @app.get("/redoc", include_in_schema=False)
    async def redoc_html():
        return get_redoc_html(
            openapi_url=app.openapi_url,
            title=app.title + " - ReDoc",
            redoc_js_url="/static/redoc.standalone.js",
        )

    @app.get(app.swagger_ui_oauth2_redirect_url, include_in_schema=False)
    async def swagger_ui_redirect():
        return get_swagger_ui_oauth2_redirect_html()


def get_logs(app):
    @app.on_event("startup")
    async def startup_event():
        "设置访问日志格式 增加访问时间的记录"
        logger = logging.getLogger("uvicorn.access")
        # console_formatter = uvicorn.logging.ColourizedFormatter(
        #     "{asctime} {levelprefix} : {message}",
        #     style="{", use_colors=True)
        console_formatter = uvicorn.logging.ColourizedFormatter(
            "{asctime} : {message}",
            style="{", use_colors=True)
        logger.handlers[0].setFormatter(console_formatter)