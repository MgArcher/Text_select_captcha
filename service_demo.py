# !/usr/bin/env python
# -*-coding:utf-8 -*-

"""
# File       : service_demo.py.py
# Time       ：2023/11/14 17:10
# Author     ：yujia
# version    ：python 3.6
# Description：
"""
import base64
import requests
import json


url = "http://localhost:8000/dianxuan/identify"

image_path = "docs/res.jpg"
with open(image_path, 'rb') as f:
    t = f.read()

data = {
  "dataType": 2,
  "imageSource": base64.b64encode(t).decode('utf-8'),
  # "imageID": "string"
}

print(json.dumps(data, ensure_ascii=False, indent=4))
response = requests.post(url, json=data)
print(response.text)

data = json.dumps(data)
response = requests.post(url, data=data)
print(response.text)
