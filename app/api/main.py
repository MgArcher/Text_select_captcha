from fastapi import APIRouter

from app.api.routes import dianxuan
from app.api import swagger

api_router = APIRouter()
api_router.include_router(swagger.app)
api_router.include_router(dianxuan.router, prefix="/dianxuan", tags=["点选验证码识别"])

