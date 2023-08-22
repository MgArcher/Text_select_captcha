# !/usr/bin/env python
# -*-coding:utf-8 -*-

"""
# File       : __init__.py.py
# Time       ：2021/9/26 14:11
# Author     ：yujia
# version    ：python 3.6
# Description：
"""
from fastapi import FastAPI

from app.utils.res_api import Api
from fastapi.middleware.cors import CORSMiddleware
from app.utils.docs import swagger

from app.api.clickOn import ClickOn
from app.api.clickOnFile import ClickOnFile


def create_app(app):
    # 接口文档
    swagger(app)
    # 配置跨域请求
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_methods=["*"],
        allow_headers=["*"],
    )
    api = Api(app)
    add_api(api)
    return app


def add_api(api):
    api.add_resource(ClickOn(), "/clickOn", tages=["识别"], summary="返回识别结果")
    api.add_resource(ClickOnFile(), "/clickOnFile", tages=["识别"], summary="返回识别结果图片")


app = FastAPI(
    title="验证码识别",
    description="验证码识别",
    version="1.1.10",
    docs_url=None, redoc_url=None
)
app = create_app(app)
print('后台程序启动成功...')