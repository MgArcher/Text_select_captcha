import os
from fastapi import FastAPI
from starlette.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles

from app.api.main import api_router


app = FastAPI(
    title="验证码识别",
    description="验证码识别",
    version="1.1.10",
    docs_url=None, redoc_url=None
)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(api_router)

# 挂载swagger静态页面
app.mount("/static", StaticFiles(directory=f"{os.path.dirname(__file__)}/swagger"), name="static")