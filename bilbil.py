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


class BilBil(object):
    def __init__(self):
        chrome_options = self.options()
        self.browser = webdriver.Chrome(chrome_options=chrome_options)
        # self.browser.maximize_window()
        self.wait = WebDriverWait(self.browser, 30)
        self.url = "https://passport.bilibili.com/login"
        self.cap = captcha.TextSelectCaptcha()

    # def __del__(self):
    #     self.browser.close()

    def options(self):
        chrome_options = webdriver.ChromeOptions()
        return chrome_options

    def click(self, xpath):
        self.wait.until(EC.presence_of_element_located(
            (By.XPATH, xpath))).click()

    def get_location(self, element):
        # 获取元素在屏幕上的位置信息
        location = element.location
        size = element.size
        height = size['height']
        width = size['width']
        left = location['x']
        top = location['y']
        right = left + width
        bottom = top + height
        script = f"return {{'left': {left}, 'top': {top}, 'right': {right}, 'bottom': {bottom}}};"
        rect = self.browser.execute_script(script)

        # # 计算元素的中心坐标
        # center_x = int((rect['left'] + rect['right']) / 2)
        # center_y = int((rect['top'] + rect['bottom']) / 2)
        # # 计算元素左上
        center_x = int(rect['left'])
        center_y = int(rect['top'])
        return center_x, center_y

    def bibi(self):
        url = "https://passport.bilibili.com/login"
        self.browser.get(url)
        xpath = '//*[@id="app"]/div[2]/div[2]/div[3]/div[2]/div[1]/div[1]/input'
        self.wait.until(EC.presence_of_element_located(
            (By.XPATH, xpath))).send_keys('Python')

        xpath = '//*[@id="app"]/div[2]/div[2]/div[3]/div[2]/div[1]/div[3]/input'
        self.wait.until(EC.presence_of_element_located(
            (By.XPATH, xpath))).send_keys('Python')
        xpath = '//*[@id="app"]/div[2]/div[2]/div[3]/div[2]/div[2]/div[2]'
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
            content = requests.get(url).content
            #送入模型识别
            plan = self.cap.run(content)
            # 获取验证码坐标
            X, Y = self.get_location(logo)
            # 前端展示对于原图的缩放比例
            lan_x = 306/334
            lan_y = 343/384
            for crop in plan:
                x1, y1, x2, y2 = crop
                x, y = [(x1 + x2) / 2, (y1 + y2) / 2]
                print(x, y)
                ActionChains(self.browser).move_by_offset(X + x*lan_x, Y + y*lan_y).click().perform()
                ActionChains(self.browser).move_by_offset(-(X + x*lan_x), -(Y + y*lan_y)).perform()  # 将鼠标位置恢复到移动前
                time.sleep(0.5)
            xpath = "/html/body/div[4]/div[2]/div[6]/div/div/div[3]/a/div"
            self.click(xpath)
        else:
            print("error: 未获得到验证码地址")
            # draw(content, res)
            content = None

        return content


if __name__ == '__main__':
    jd = BilBil()
    jd.bibi()

