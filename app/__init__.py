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
from fastapi_restful import Api

from app.api.clickOn import ClickOn
from app.api.clickOnFile import ClickOnFile
from app.api.clickOnForm import ClickOnForm
from app.api.clickonGat import ClickOnGat
from fastapi.middleware.cors import CORSMiddleware

from app.utils.logs_docs import get_docs

def create_app():
    app = FastAPI()
    # 关闭文档
    # app = FastAPI(docs_url=None, redoc_url=None)
    get_docs(app)
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
    api.add_resource(ClickOn(), "/clickOn")
    api.add_resource(ClickOnFile(), "/clickOnFile")
    api.add_resource(ClickOnForm(), "/clickOnForm")
    api.add_resource(ClickOnGat(), "/clickOnGat")