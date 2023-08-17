# !/usr/bin/env python
# -*-coding:utf-8 -*-

"""
# File       : load.py
# Time       ：2023/8/17 17:32
# Author     ：yujia
# version    ：python 3.6
# Description：
"""
import zlib
from cryptography.fernet import Fernet


def decryption(path):
    key = b"KjKZJXo_tUPjZVIO896n15t8U5xjXsoRbGLGKnLDb08="
    f = Fernet(key)
    with open(path, 'rb') as fr:
        onnx_file = fr.read()
    onnx_file = zlib.decompress(onnx_file)
    onnx_file = f.decrypt(onnx_file)
    return onnx_file