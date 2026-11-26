# !/usr/bin/env python
# -*-coding:utf-8 -*-

"""响应结构说明（给文档和读代码的人看）。

刻意不挂 response_model：那会把可选字段补成 null（例如 warning），等于悄悄改契约。
"""
from __future__ import annotations

from typing import Any, Optional

from pydantic import BaseModel, Field


class LegacyRes(BaseModel):
    """旧字段（v7），两个列表都按点击顺序排列。"""

    imgW: int
    imgH: int
    point: list[dict[str, float]]
    corp: list[dict[str, int]]


class IdentifyData(BaseModel):
    """`POST /api/v1/identify` 的 data：新旧字段同一层，来自同一次推理。"""

    imageID: str = ""
    scheme: str

    res: LegacyRes

    boxes: list[list[float]] = Field(default_factory=list)
    points: list[list[float]] = Field(default_factory=list)
    scores: list[float] = Field(default_factory=list)
    order: list[int] = Field(default_factory=list)
    margin: float = 0.0
    n_prompts: int = 0
    n_targets: int = 0
    timings_ms: dict[str, float] = Field(default_factory=dict)
    error: Optional[str] = None
    warning: Optional[str] = None


class HealthResponse(BaseModel):
    status: str
    loaded: list[str] = Field(default_factory=list)
    errors: dict[str, str] = Field(default_factory=dict)
    scheme: str
    spec: Optional[dict[str, Any]] = None


class SchemeInfo(BaseModel):
    name: str
    title: str
    description: str
    loaded: bool
    ready: bool
    missing: list[str] = Field(default_factory=list)
