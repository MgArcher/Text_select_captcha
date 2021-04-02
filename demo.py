"""
@author: jiajia
@file: demo.py
@time: 2021/3/28 15:31
"""
from src import captcha
from drawing import draw
import time

if __name__ == '__main__':
    path = r"img/1234.jpg"
    with open(path, 'rb') as f:
        path = f.read()
    cap = captcha.TextSelectCaptcha(GPU=True)
    s1 = time.time()
    data = cap.run(path)
    print(data)
    print(time.time() - s1)
    draw(path, data)
