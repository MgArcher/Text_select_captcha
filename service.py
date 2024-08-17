# !/usr/bin/env python
# -*-coding:utf-8 -*-

"""
# File       : service.py
# Time       ：2021/9/26 14:36
# Author     ：yujia
# version    ：python 3.6
# Description：
"""
import uvicorn
from app import main

if __name__ == '__main__':
    uvicorn.run(main.app, host="127.0.0.1", port=8000)