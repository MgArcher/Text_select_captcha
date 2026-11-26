#!/usr/bin/env python
# -*-coding:utf-8 -*-

"""B 站登录 + 极验文字点选验证码（v8 引擎）。

    python bilbil.py -u <账号> -p <密码> [--headless] [--retries 3]

退出码：0 登录成功；2 验证码通过但账号/密码不对；1 其他失败。
选择器用 placeholder / class 等语义定位（绝对 XPath 一改版就失效），改版只改下面常量。
"""
from __future__ import annotations

import argparse
import re
import sys
import time
from typing import Optional
from urllib.request import Request, urlopen

from playwright.sync_api import Locator, Page, sync_playwright

from src import V8Pipeline

URL = "https://passport.bilibili.com/login"
ACCOUNT = ('input[placeholder="请输入账号"]', 'input[placeholder*="账号"]', 'input[type="text"]')
PASSWORD = ('input[placeholder="请输入密码"]', 'input[type="password"]')
SUBMIT = ("div.btn_primary", 'button:has-text("登录")')
STEALTH = "Object.defineProperty(navigator,'webdriver',{get:()=>false});"
BAD_CRED = ("账号或密码", "密码错误", "账号不存在", "用户名或密码", "登录失败")

# 极验面板状态：成功后面板隐藏、tip 变「验证成功」；失败时面板仍在、tip 为失败文案
STATE_JS = """
() => {
  const p = document.querySelector('.geetest_panel');
  const b = p ? p.getBoundingClientRect() : null;
  const s = p ? getComputedStyle(p) : null;
  const open = !!(b && b.width > 0 && b.height > 0 && s.display !== 'none' && s.visibility !== 'hidden');
  const t = document.querySelector('.geetest_result_tip');
  return {open, tip: t ? (t.innerText || '').trim() : ''};
}
"""

_pipe: Optional[V8Pipeline] = None


def pipe() -> V8Pipeline:
    global _pipe
    if _pipe is None:
        _pipe = V8Pipeline()
    return _pipe


def first(page: Page, selectors, timeout: float = 5000) -> Optional[Locator]:
    """按顺序找一个可见元素，全失败返回 None。"""
    for sel in selectors:
        loc = page.locator(sel).first
        try:
            loc.wait_for(state="visible", timeout=timeout)
            return loc
        except Exception:
            continue
    return None


def bg_url(page: Page) -> Optional[str]:
    """验证码背景图挂在 .geetest_item_wrap 上。"""
    style = page.evaluate(
        "() => {const e = document.querySelector('.geetest_item_wrap');"
        "return e ? e.getAttribute('style') || '' : '';}"
    )
    match = re.search(r'url\("(.+?)"\)', style)
    return match.group(1) if match else None


def wait_result(page: Page, ignore: str = "", timeout: float = 8.0) -> Optional[bool]:
    """面板关闭 = 通过。

    `ignore` 是提交前的旧提示：刷新验证码后极验会沿用上一轮文案，
    不排除掉就会把上一轮的「验证失败」当成本次结果（实测会误判成功）。
    """
    end = time.time() + timeout
    while time.time() < end:
        state = page.evaluate(STATE_JS)
        if not state["open"]:
            return True
        tip = state["tip"]
        if tip and tip != ignore:
            if "成功" in tip or "通过" in tip:
                return True
            if "失败" in tip or "错误" in tip or "重新" in tip:
                return False
        page.wait_for_timeout(200)
    return None


def refresh(page: Page, old_url: str) -> None:
    """点刷新，并等图片真的换了 —— 否则等于对着同一张图重复失败。"""
    try:
        page.locator("a.geetest_refresh").first.click(timeout=2000)
    except Exception:
        return
    for _ in range(20):
        page.wait_for_timeout(300)
        if bg_url(page) != old_url:
            page.wait_for_timeout(500)
            return


def click_captcha(page: Page, tries: int = 3, gap: float = 0.8) -> bool:
    """背景图喂给 v8，按 wrap 的显示宽度换算坐标后依次点击。

    background-size:100% 是等比缩放，x/y 用同一个比例，基准是 wrap 本身；
    .geetest_item_img 只是 height:112% 的遮罩，拿它算会让 y 偏 12%。
    """
    for _ in range(tries):
        url = bg_url(page)
        print(f"url:{url}")
        if not url:
            page.wait_for_timeout(800)
            continue

        raw = urlopen(Request(url, headers={"User-Agent": "Mozilla/5.0"}), timeout=10).read()
        plan = pipe().predict(raw)
        box = page.locator(".geetest_item_wrap").first.bounding_box()
        if plan.error or not plan.boxes or not box:
            refresh(page, url)
            continue

        scale = box["width"] / plan.img_w
        print(f"点数={len(plan.boxes)} margin={plan.margin:.3f} scale={scale:.4f}")
        for px, py in plan.points:
            x, y = box["x"] + px * scale, box["y"] + py * scale
            page.mouse.move(x, y)
            page.mouse.click(x, y)
            time.sleep(gap)
        time.sleep(2)
        commit_cls = page.evaluate(
            "() => {const c = document.querySelector('a.geetest_commit'); return c ? c.className : ''}"
        )
        if "disable" in commit_cls:  # 点的字数不够，"确认"根本没启用
            print("点数不足，确认按钮未启用，换一张")
            refresh(page, url)
            continue

        before = page.evaluate(STATE_JS)["tip"]
        page.locator("a.geetest_commit").first.click(timeout=3000)
        if wait_result(page, ignore=before) is True:
            print("点选验证通过")
            return True
        print("点选验证失败，换一张")
        refresh(page, url)
    return False


def handle_captcha(page: Page, wait: float = 8.0) -> Optional[bool]:
    """True/False = 验证码结果；None = 没出现验证码。"""
    for _ in range(int(wait / 0.3)):
        if page.locator(".geetest_item_wrap").count():
            print("检测到点选验证码")
            return click_captcha(page)
        page.wait_for_timeout(300)
    return None


def login(page: Page, user: str, pwd: str) -> str:
    """返回 logged_in / bad_credentials / captcha_failed / no_form / unknown。"""
    account, password = first(page, ACCOUNT, 8000), first(page, PASSWORD)
    if account is None or password is None:
        return "no_form"
    account.fill(user)
    password.fill(pwd)

    submit = first(page, SUBMIT)
    if submit is None:
        return "no_form"
    for _ in range(25):  # 填完账号密码按钮才从 disabled 变可用
        if "disabled" not in (submit.get_attribute("class") or ""):
            break
        page.wait_for_timeout(200)
    submit.click()

    passed = handle_captcha(page)
    for _ in range(24):
        page.wait_for_timeout(500)
        if logged_in(page):
            return "logged_in"
        if passed is not False:
            text = page.evaluate("() => (document.body.innerText || '').slice(0, 6000)")
            hint = next((h for h in BAD_CRED if h in text), None)
            if hint:
                print(f"验证码已通过 ✅（{hint}）" if passed else f"未出现验证码（{hint}）")
                return "bad_credentials"
    return "captcha_failed" if passed is False else "unknown"


def logged_in(page: Page) -> bool:
    """登录成功会下发 DedeUserID cookie，比查 DOM 文案稳。"""
    return any(c.get("name") == "DedeUserID" for c in page.context.cookies())


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="B 站登录 + 极验文字点选（v8 引擎）")
    p.add_argument("-u", "--username", default="python")
    p.add_argument("-p", "--password", default="python")
    p.add_argument("--headless", action="store_true")
    p.add_argument("--retries", type=int, default=3, help="整体重试次数")
    return p.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)

    with sync_playwright() as pw:
        browser = pw.chromium.launch(headless=args.headless)
        context = browser.new_context(locale="zh-CN", viewport={"width": 1440, "height": 900})
        context.add_init_script(STEALTH)
        page = context.new_page()
        page.set_default_timeout(10_000)
        try:
            for i in range(1, args.retries + 1):
                print(f"第 {i}/{args.retries} 次尝试")
                try:
                    page.goto(URL, wait_until="domcontentloaded", timeout=60_000)
                    page.wait_for_timeout(2500)
                    status = login(page, args.username, args.password)
                except Exception as exc:
                    print("异常:", exc)
                    status = "error"

                if status == "logged_in":
                    print("登录成功")
                    return 0
                if status == "bad_credentials":
                    print("登录失败：账号/密码不对")
                    return 2
                print("未成功:", status)
                page.wait_for_timeout(2000)
        finally:
            context.close()
            browser.close()
    return 1


if __name__ == "__main__":
    sys.exit(main())
