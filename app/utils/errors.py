# -*- coding: utf-8 -*-
from fastapi.exceptions import RequestValidationError
from starlette.responses import PlainTextResponse, JSONResponse
from starlette.requests import Request
from starlette.status import HTTP_401_UNAUTHORIZED, HTTP_403_FORBIDDEN, HTTP_422_UNPROCESSABLE_ENTITY
from fastapi.encoders import jsonable_encoder


def bad_request(message="参数内容不得为空！"):
    response = {'code': 400, 'msg': "请求参数内容错误", "data": message}
    response.status_code = 400
    return response


def bad_error(message=''):
    response = {'code': 400, 'msg': "系统错误",  "data": message}
    return response


async def request_validation_exception_handler(
        request: Request, exc: RequestValidationError
) -> JSONResponse:
    """
    捕捉422报错并进行自定义处理
    :param request:
    :param exc:
    :return:
    """
    return JSONResponse(
        status_code=HTTP_422_UNPROCESSABLE_ENTITY,
        content={'code': 422, 'msg': "请求参数格式错误！", "data": jsonable_encoder(exc.errors())}
    )
