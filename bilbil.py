#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@author: jiajia
@file: bilbil.py
@time: 2020/8/22 18:48
"""
import re
import time
import base64

import requests
from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.webdriver.common.action_chains import ActionChains

from src import captcha
from drawing import draw


# 初始化项目
cap = captcha.TextSelectCaptcha()


def to_selenium(res):
    place = []
    title = [i['content'] for i in res if i['classes'] == "title"][0]
    for t in title:
        for item in res:
            if item['classes'] == "target":
                x1, y1, x2, y2 = item['crop']
                if item['content'] == t:
                    place.append(
                        {
                            "text": t,
                            "place": [(x1 + x2)/2, (y1 + y2)/2]
                        }
                    )
    return place


class BilBil(object):
    def __init__(self):
        chrome_options = self.options()
        self.browser = webdriver.Chrome(chrome_options=chrome_options)
        # self.browser.maximize_window()
        self.wait = WebDriverWait(self.browser, 30)
        self.url = "https://passport.bilibili.com/login"
        self.x_offset = 45

    def __del__(self):
        self.browser.close()

    def options(self):
        chrome_options = webdriver.ChromeOptions()
        return chrome_options

    def click(self, xpath):
        self.wait.until(EC.presence_of_element_located(
            (By.XPATH, xpath))).click()

    def bibi(self):
        url = "https://passport.bilibili.com/login"
        self.browser.get(url)
        xpath = '//*[@id="login-username"]'
        self.wait.until(EC.presence_of_element_located(
            (By.XPATH, xpath))).send_keys('Python')
        xpath = '//*[@id="login-passwd"]'
        self.wait.until(EC.presence_of_element_located(
            (By.XPATH, xpath))).send_keys('Python')
        xpath = '//*[@id="geetest-wrap"]//*[@class="btn btn-login"]'
        self.click(xpath)

        xpath = '//*[@class="geetest_item_img"]'
        logo = self.wait.until(EC.presence_of_element_located(
        (By.XPATH, xpath)))
        # 获取图片路径
        f = logo.get_attribute('src')
        if not f:
            return None
        content = requests.get(f).content
        res = cap.run(content)
        plan = to_selenium(res)
        X, Y = logo.location['x'], logo.location['y']
        # print(X, Y)
        lan_x = 259/334
        lan_y = 290/384
        for p in plan:
            x, y = p['place']

            ActionChains(self.browser).move_by_offset(X + x*lan_x - self.x_offset, Y + y*lan_y).click().perform()
            ActionChains(self.browser).move_by_offset(-(X + x*lan_x - self.x_offset), -(Y + y*lan_y)).perform()  # 将鼠标位置恢复到移动前
            time.sleep(0.5)
        xpath = "//*[@class='geetest_commit_tip']"
        self.click(xpath)

        # draw(content, res)
        time.sleep(100)


if __name__ == '__main__':
    jd = BilBil()
    jd.bibi()