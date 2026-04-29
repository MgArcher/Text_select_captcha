# !/usr/bin/env python
# -*-coding:utf-8 -*-

"""
# File       : output.py.py
# Time       ：2026/4/29 15:52
# Author     ：yujia
# version    ：python 3.6
# Description：
"""

from pydantic import BaseModel, Field
from typing import Optional

from pydantic import BaseModel, Field
from typing import Optional

class Input(BaseModel):
    dataType: int = Field(
        default=1,
        example=1,
        description="文件类型：1:链接地址，2:文件字节流转base64(base64不含头)"
    )
    imageSource: str = Field(
        default="https://static.geetest.com/captcha_v3/batch/v3/140155/2026-04-29T15/word/474b661c0cd74331b14937fef099eb19.jpg?challenge=894235a05cd62c9ec2e9c6ae5fdb4e22",
        example="https://static.geetest.com/captcha_v3/batch/v3/140155/2026-04-29T15/word/474b661c0cd74331b14937fef099eb19.jpg?challenge=894235a05cd62c9ec2e9c6ae5fdb4e22",
        description="源文件地址或 base64 编码的字节流"
    )
    imageID: Optional[str] = Field(
        default=None,
        example="string",
        description="图片名称或 id"
    )

