from fastapi import APIRouter
from app.api.endpoints import dianxuan

router = APIRouter()
router.include_router(dianxuan.router, tags=["识别"])
