from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse
from app.api.router import router as v1_router
from app.utils.errors import BadRequestError, ServiceError

def create_app() -> FastAPI:
    app = FastAPI(title="文字点选验证码识别服务", version="1.0.0")

    @app.exception_handler(BadRequestError)
    async def bad_request_handler(request: Request, exc: BadRequestError):
        return JSONResponse(status_code=400, content={"code": 400, "msg": exc.detail, "data": None})

    @app.exception_handler(ServiceError)
    async def service_error_handler(request: Request, exc: ServiceError):
        return JSONResponse(status_code=500, content={"code": 500, "msg": exc.detail, "data": None})

    @app.get("/")
    def root():
        return {"code": 200, "msg": "成功", "data": "ok"}

    app.include_router(v1_router, prefix="/api/v1")
    return app

app = create_app()  # 供 uvicorn 直接导入