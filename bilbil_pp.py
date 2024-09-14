# !/usr/bin/env python
# -*-coding:utf-8 -*-

"""
# File       : bilbil2.py
# Time       ：2024/9/12 19:04
# Author     ：yujia
# version    ：python 3.6
# Description：
"""
import requests
import re
import asyncio
from src import captcha
from pyppeteer import launch

URL = "https://passport.bilibili.com/login"
cap = captcha.TextSelectCaptcha()


async def verify(url):
    content = requests.get(url).content
    # 送入模型识别
    plan = cap.run(content)
    return plan


async def log_console_message(msg):
    # 获取 console.log 输出的对象
    for handle in msg.args:
        value = await handle.jsonValue()
        print(f'Console: {value}')


async def init(page):
    # 监听控制台输出
    page.on('console', lambda msg: asyncio.ensure_future(log_console_message(msg)))
    # 拦截鼠标点击事件并输出坐标
    await page.evaluate('''() => {  
                document.addEventListener('click', (event) => {  
                    console.log({  
                        x: event.clientX,  
                        y: event.clientY,  
                        target: event.target.tagName,  
                        classList: event.target.classList.toString()  
                    });  
                });  
            }''')
    await page.evaluate(
        '''() =>{ Object.defineProperties(navigator,{ webdriver:{ get: () => false } }) }''')


async def login(page):
    xpath = '//*[@id="app-main"]/div/div[2]/div[3]/div[2]/div[1]/div[1]/input'
    elements = await page.xpath(xpath)
    await elements[0].type('python')
    xpath = '//*[@id="app-main"]/div/div[2]/div[3]/div[2]/div[1]/div[3]/input'
    elements = await page.xpath(xpath)
    await elements[0].type('python')
    xpath = '//*[@id="app-main"]/div/div[2]/div[3]/div[2]/div[2]/div[2]'
    elements = await page.xpath(xpath)
    await elements[0].click()
    xpath = '//*[@class="geetest_item_wrap"]'
    element = await page.waitForXPath(xpath)
    await page.waitForFunction('''() => {  
        const element = document.querySelector('[class="geetest_item_wrap"]');  
        return element && element.getAttribute("style") !== null;  
    }''')
    style_content = await page.evaluate('(element) => element.getAttribute("style")', element)
    url = re.findall('url\("(.+?)"\);', style_content) if style_content else None
    if url:
        plan = await verify(url[0])
    else:
        print("未获取到图片")
        return
    bounding_box = await page.evaluate('''(element) => {
    const
    rect = element.getBoundingClientRect();
    return {
        x: rect.left,
        y: rect.top,
        width: rect.width,
        height: rect.height
    };
    }''', element)
    print(f"识别位置信息: {plan}")
    print(f"元素位置信息: {bounding_box}")
    X, Y = bounding_box['x'], bounding_box['y']
    lan_x = 306 / 344
    lan_y = 343 / 384
    for crop in plan:
        x1, y1, x2, y2 = crop
        x, y = [(x1 + x2) / 2, (y1 + y2) / 2]
        x, y = X + x * lan_x, Y + y * lan_y
        await page.mouse.click(x, y)  # 在坐标 (100, 100) 点击
        await asyncio.sleep(0.5)

    xpath = "/html/body/div[4]/div[2]/div[6]/div/div/div[3]/a/div"
    elements = await page.xpath(xpath)
    await elements[0].click()


async def main():
    browser = await launch(headless=False)
    page = await browser.newPage()
    await page.goto(URL)
    await init(page)
    await login(page)
    await asyncio.sleep(1000)


if __name__ == '__main__':
    asyncio.get_event_loop().run_until_complete(main())