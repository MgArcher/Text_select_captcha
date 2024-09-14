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

cap = captcha.TextSelectCaptcha()
def verify(url):
    content = requests.get(url).content
    # 送入模型识别
    plan = cap.run(content)
    return plan


class BilBil(object):
    def __init__(self):
        chrome_options = self.options()
        self.browser = webdriver.Chrome(options=chrome_options)
        # self.browser.maximize_window()
        self.wait = WebDriverWait(self.browser, 30)
        self.url = "https://passport.bilibili.com/login"

    # def __del__(self):
    #     self.browser.close()

    def options(self):
        chrome_options = webdriver.ChromeOptions()
        return chrome_options

    def click(self, xpath):
        self.wait.until(EC.presence_of_element_located(
            (By.XPATH, xpath))).click()

    def get_location(self, element):
        rect = element.rect
        center_x = int(rect['x'] - 50)
        center_y = int(rect['y'])
        return center_x, center_y

    def bibi(self):
        url = "https://passport.bilibili.com/login"
        self.browser.get(url)
        # xpath = '//*[@id="app"]/div[2]/div[2]/div[3]/div[2]/div[1]/div[1]/input'
        xpath = '//*[@id="app-main"]/div/div[2]/div[3]/div[2]/div[1]/div[1]/input'
        self.wait.until(EC.presence_of_element_located(
            (By.XPATH, xpath))).send_keys('Python')

        # xpath = '//*[@id="app"]/div[2]/div[2]/div[3]/div[2]/div[1]/div[3]/input'
        xpath = '//*[@id="app-main"]/div/div[2]/div[3]/div[2]/div[1]/div[3]/input'
        self.wait.until(EC.presence_of_element_located(
            (By.XPATH, xpath))).send_keys('Python')
        # xpath = '//*[@id="app"]/div[2]/div[2]/div[3]/div[2]/div[2]/div[2]'
        xpath = '//*[@id="app-main"]/div/div[2]/div[3]/div[2]/div[2]/div[2]'
        self.click(xpath)

        time.sleep(2)
        xpath = '//*[@class="geetest_item_wrap"]'
        logo = self.wait.until(EC.presence_of_element_located(
        (By.XPATH, xpath)))
        # 获取图片路径
        f = logo.get_attribute('style')
        url = re.findall('url\("(.+?)"\);', f)
        if url:
            url = url[0]
            print(url)
            #送入模型识别
            plan = verify(url)
            # 获取验证码坐标
            X, Y = self.get_location(logo)
            print(X, Y)
            # 前端展示对于原图的缩放比例
            # 306 * 343
            # 344 *384
            lan_x = 306/344
            lan_y = 343/384
            # lan_x = lan_y = 1
            # ActionChains(self.browser).move_by_offset(X, Y).click().perform()
            # time.sleep(11111)
            for crop in plan:
                x1, y1, x2, y2 = crop
                x, y = [(x1 + x2) / 2, (y1 + y2) / 2]
                print(x, y)
                ActionChains(self.browser).move_by_offset(X + x*lan_x, Y + y*lan_y).click().perform()
                ActionChains(self.browser).move_by_offset(-(X + x*lan_x), -(Y + y*lan_y)).perform()  # 将鼠标位置恢复到移动前
                time.sleep(0.5)
            # time.sleep(1000)
            xpath = "/html/body/div[4]/div[2]/div[6]/div/div/div[3]/a/div"
            self.click(xpath)
            time.sleep(1000)
            try:
                time.sleep(1)
                xpath = "/html/body/div[4]/div[2]/div[6]/div/div/div[3]/div/a[2]"
                self.click(xpath)
                sign = False
            except:
                sign = True
        else:
            print("error: 未获得到验证码地址")
            # draw(content, res)
            sign = False

        return sign


if __name__ == '__main__':
    jd = BilBil()
    s = jd.bibi()