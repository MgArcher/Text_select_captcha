"""
@author: jiajia
@file: demo.py
@time: 2021/3/28 15:31
"""
from src import captcha
from drawing import draw
import time
import json

if __name__ == '__main__':
    path = r"domo.jpg"
    cap = captcha.TextSelectCaptcha()
    s1 = time.time()
    data = cap.run(path)
    s2 = time.time()
    # draw(path, data)
    data = json.dumps(data, sort_keys=True, indent=4, ensure_ascii=False)
    print(data)
    print(s2 - s1)
