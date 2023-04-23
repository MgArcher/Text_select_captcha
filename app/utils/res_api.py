# !/usr/bin/env python
# -*-coding:utf-8 -*-

"""
# File       : res_api.py
# Time       ：2021/9/30 13:49
# Author     ：yujia
# version    ：python 3.6
# Description：
"""
from fastapi_restful.cbv_base import Resource, APIRouter, Any, _cbv
from fastapi_restful import Api as api


class Api(api):
    """重写add_resource函数"""
    def add_resource(self, resource: Resource, *urls: str, **kwargs: Any) -> None:
        router = APIRouter()
        _cbv(router, type(resource), *urls, instance=resource)
        router.routes[0].tags = kwargs.get("tages")
        router.routes[0].summary = kwargs.get("summary")
        op = hasattr(resource, 'output_model')
        if op:
            router.routes[0].response_model = resource.output_model
        self.app.include_router(router)
