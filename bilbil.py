#!/usr/bin/env python
# -*-coding:utf-8 -*-

"""
# File       : bilbil2_playwright_sync.py
# Time       ：根据原文件转换（同步版本）
# Author     ：yujia
# version    ：python 3.7+
# Description：使用 playwright 同步 API 重写的 B 站登录脚本
"""
from PIL import Image
import io
import re
import time
import requests
from playwright.sync_api import sync_playwright
from src import captcha  # 请确保该模块存在，并提供 TextSelectCaptcha 类

cap = captcha.TextSelectCaptcha()
URL = "https://passport.bilibili.com/login"


def log_console_message(msg):
    """监听控制台输出，打印消息（同步回调）"""
    text = msg.text
    if text:
        print(f'Console: {text}')


def init(page):
    """初始化页面：监听控制台、修改 webdriver 属性、注入鼠标点击记录"""
    # 监听 console 输出
    page.on('console', log_console_message)

    # 注入鼠标点击事件记录（与原始功能一致）
    page.evaluate('''() => {
        document.addEventListener('click', (event) => {
            console.log({
                x: event.clientX,
                y: event.clientY,
                target: event.target.tagName,
                classList: event.target.classList.toString()
            });
        });
    }''')

    # 隐藏 webdriver 特征
    page.add_init_script('''() => {
        Object.defineProperty(navigator, 'webdriver', { get: () => false });
    }''')


def handle_captcha(page):
    """
    处理 B 站登录页的极验点选验证码
    返回: True 处理成功，False 失败
    """
    try:
        # 等待验证码容器和图片元素
        captcha_element = page.wait_for_selector('//*[@class="geetest_item_wrap"]', timeout=5000)
        img_element = page.wait_for_selector('//*[@class="geetest_item_img"]', timeout=5000)
        # 等待背景图加载（style 属性非空）
        page.wait_for_function(
            '''element => element && element.getAttribute("style") !== null''',
            arg=captcha_element
        )

        # 提取背景图片 URL
        style_content = captcha_element.get_attribute('style')
        url_match = re.search(r'url\("(.+?)"\);', style_content) if style_content else None
        if not url_match:
            print("未获取到图片 URL")
            return False
        img_url = url_match.group(1)
        print(f"验证码图片 URL: {img_url}")

        # 调用识别模型，获得点击目标（相对原图的边界框）
        # plan, orig_w, orig_h = verify(img_url)  # verify 返回 (plan, width, height)
        resp = requests.get(img_url)
        content = resp.content
        plan = cap.run_dict(content)  # 模型返回的 crop 坐标列表，格式 [(x1,y1,x2,y2), ...]

        orig_w, orig_h = plan.get("imgW"), plan.get("imgH")
        print(f"图片尺寸: {orig_w} x {orig_h}")


        if not plan:
            print("未获取到点击计划")
            return False

        # 获取网页中验证码元素的显示位置和尺寸
        box = captcha_element.bounding_box()
        img_box = img_element.bounding_box()
        if not box or not img_box:
            print("无法获取验证码元素位置")
            return False
        X, Y = box['x'], box['y']
        display_w, display_h = img_box['width'], img_box['height']
        print(f"显示尺寸: {display_w} x {display_h}, 原始尺寸: {orig_w} x {orig_h}")

        scale_x = display_w / orig_w
        scale_y = display_h / orig_h
        time.sleep(1)
        # 依次点击每个目标
        for point in plan.get("point"):
            x_rel = point.get("x_rel")
            y_rel = point.get("y_rel")
            # 映射到页面实际坐标
            click_x = X + x_rel * scale_x
            click_y = Y + y_rel * scale_y
            page.mouse.click(click_x, click_y)
            time.sleep(0.8)  # 避免点击频率过快被风控

        # 点击“确认”按钮
        confirm_btn = page.locator('//*[@class="geetest_commit_tip"]')
        confirm_btn.click()
        print("验证码点击完成，已提交")
        return True

    except Exception as e:
        print(f"验证码处理异常: {e}")
        return False


def login(page):
    """执行登录操作：填写账号密码、点击登录、处理验证码"""
    # 填写账号
    page.locator('//*[@id="app-main"]/div/div[2]/div[3]/div[2]/div[1]/div[1]/input').fill('python')
    # 填写密码
    page.locator('//*[@id="app-main"]/div/div[2]/div[3]/div[2]/div[1]/div[3]/input').fill('python')
    # 点击登录按钮
    page.locator('//*[@id="app-main"]/div/div[2]/div[3]/div[2]/div[2]/div[2]').click()

    try:
        page.wait_for_selector('//*[@class="geetest_item_wrap"]', timeout=2000)
        print("验证码已出现，开始处理...")
        if handle_captcha(page):
            print("验证码处理完成")
        else:
            print("验证码处理失败")
    except Exception:
        print("未检测到验证码，可能登录成功或无需验证")

    time.sleep(2)
    try:
        # 当检测不到页面有验证码元素认为成功了
        wrap = page.wait_for_selector('//*[@class="geetest_item_wrap"]', timeout=500)
        print("识别失败！")
    except Exception:
        print("识别成功！")


def main():
    with sync_playwright() as p:
        # 启动浏览器（headless=False 显示窗口，便于调试）
        browser = p.chromium.launch(headless=False)
        context = browser.new_context()
        page = context.new_page()
        page.goto(URL)
        init(page)
        login(page)
        # 保留页面一段时间，观察结果
        time.sleep(1000)


if __name__ == '__main__':
    main()