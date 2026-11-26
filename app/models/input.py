# !/usr/bin/env python
# -*-coding:utf-8 -*-

"""请求模型（保持历史字段）。

没有 `inputChars`：提示字由模型从图上检测，传提示文字既不参与推理也无法保证一致。
老调用方多传会被忽略（pydantic 默认忽略未知键）。
"""
from __future__ import annotations

from typing import Optional

from pydantic import BaseModel, Field


class Input(BaseModel):
    """识别请求体。dataType=1 -> URL；2 -> base64（可带 data: 前缀）。"""

    dataType: int = Field(1, examples=[1], description="1=URL，2=base64")
    imageSource: str = Field(
        ...,
        examples=["https://static.geetest.com/captcha_v3/batch/v3/158267/word/xxxx.jpg"],
        description="图片地址或 base64 编码的字节流",
    )
    imageID: Optional[str] = Field(
        None,
        examples=["string"],
        description="可选。传了原样回传；不传则由服务端生成并回传",
    )
