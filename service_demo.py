#!/usr/bin/env python
# -*-coding:utf-8 -*-

"""调用 Web 服务的示例客户端：图片 -> base64 -> POST /api/v1/identify -> 打印 JSON。

先起服务：

    python service.py --eager

再跑本脚本：

    python service_demo.py                                  # 默认 docs/res.jpg
    python service_demo.py --image out.jpg --service http://127.0.0.1:8002
    python service_demo.py --url https://example.com/a.jpg   # dataType=1，直接把 URL 交给服务
    python service_demo.py --raw                             # 只打印服务返回的原始 JSON

只用标准库（urllib），不需要额外装 requests。
退出码：0 识别成功；1 服务返回业务错误；2 连接失败/参数问题。
"""
from __future__ import annotations

import argparse
import base64
import json
import sys
import urllib.error
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parent
DEFAULT_IMAGE = ROOT / "docs" / "a97.jpg"
DEFAULT_SERVICE = "http://127.0.0.1:8000"
ENDPOINT = "/api/v1/identify"


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="上传 base64 图片，调用识别服务拿 JSON")
    src = p.add_mutually_exclusive_group()
    src.add_argument("--image", default=str(DEFAULT_IMAGE), help="本地图片路径")
    src.add_argument("--url", default="", help="改用图片 URL（dataType=1，服务端自己下载）")
    p.add_argument("--service", default=DEFAULT_SERVICE, help=f"服务地址，默认 {DEFAULT_SERVICE}")
    p.add_argument("--image-id", default="", help="可选，原样回传，便于对账")
    p.add_argument("--timeout", type=float, default=60.0)
    p.add_argument("--raw", action="store_true", help="只打印服务返回的原始 JSON")
    return p.parse_args(argv)


def build_payload(args: argparse.Namespace) -> dict:
    """dataType=1 传 URL；dataType=2 传 base64（把本地文件读出来编码）。"""
    if args.url:
        return {"dataType": 1, "imageSource": args.url, "imageID": args.image_id or None}

    path = Path(args.image)
    if not path.is_file():
        raise FileNotFoundError(f"图片不存在: {path}")
    raw = path.read_bytes()
    b64 = base64.b64encode(raw).decode()
    print(f"[请求] {path.name}  {len(raw)} B  ->  base64 {len(b64)} 字符")
    return {"dataType": 2, "imageSource": b64, "imageID": args.image_id or None}


def call(service: str, payload: dict, timeout: float) -> dict:
    req = urllib.request.Request(
        service.rstrip("/") + ENDPOINT,
        data=json.dumps(payload).encode(),
        headers={"Content-Type": "application/json"},
    )
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        return json.loads(resp.read())


def print_result(body: dict) -> None:
    print(f"\n[响应] code={body.get('code')}  msg={body.get('msg')}")
    data = body.get("data") or {}
    if not data:
        return

    print(f"  imageID    : {data.get('imageID')}")
    print(f"  scheme     : {data.get('scheme')}")
    print(f"  点击顺序框 : {data.get('boxes')}")
    print(f"  点击中心点 : {data.get('points')}")
    print(f"  scores     : {data.get('scores')}")
    print(f"  检测数量   : prompts={data.get('n_prompts')} targets={data.get('n_targets')}")
    print(f"  耗时(ms)   : {data.get('timings_ms')}")
    if data.get("error"):
        print(f"  error      : {data['error']}")
    if data.get("warning"):
        print(f"  warning    : {data['warning']}")

    # 旧接口字段也一并给出了，老调用方直接用 res
    res = data.get("res") or {}
    if res:
        print(f"  旧字段 res : imgW={res.get('imgW')} imgH={res.get('imgH')} "
              f"corp {len(res.get('corp') or [])} 个")


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)

    try:
        payload = build_payload(args)
    except FileNotFoundError as exc:
        print(f"[错误] {exc}")
        return 2

    try:
        body = call(args.service, payload, args.timeout)
    except urllib.error.HTTPError as exc:  # 理论上服务恒返回 200，这里兜底
        print(f"[错误] HTTP {exc.code}: {exc.read()[:300]!r}")
        return 1
    except (urllib.error.URLError, TimeoutError) as exc:
        print(f"[错误] 连不上 {args.service}: {exc}")
        print("       先启动服务：python service.py --eager")
        return 2

    if args.raw:
        print(json.dumps(body, ensure_ascii=False, indent=2))
    else:
        print_result(body)

    return 0 if body.get("code") == 200 else 1


if __name__ == "__main__":
    sys.exit(main())
