# !/usr/bin/env python
# -*-coding:utf-8 -*-

"""
# File       : demo.py
# Time       ：2026/11/3 11:20
# Description：命令行体验：跑一张验证码，输出点击顺序并可选保存标注图。

    python demo.py                               # 跑 docs/res.jpg
    python demo.py --image xx.jpg --show out.jpg # 指定图片并保存标注图
    python demo.py --json                        # 只打印 JSON
    python demo.py --detect                      # 只看检测框，不做匹配
"""
import argparse
import json

from src import V8Pipeline
from src.draw import save_annotated
from src.imageio import load_rgb


def main() -> None:
    p = argparse.ArgumentParser(description="v8 文字点选识别 demo")
    p.add_argument("--image", default="docs/res.jpg", help="输入图片（路径 / URL）")
    p.add_argument("--show", default="res_v8.jpg", help="标注图输出路径，例如 res_v8.jpg")
    p.add_argument("--config", default="v8", help="配置名或路径，默认 configs/v8.toml")
    p.add_argument("--json", action="store_true", help="只打印 JSON")
    p.add_argument("--detect", action="store_true", help="只做检测，输出原始框")
    args = p.parse_args()

    pipe = V8Pipeline(config=args.config)

    if args.detect:
        dets = pipe.detection(args.image)
        if args.json:
            print(json.dumps(dets, ensure_ascii=False, indent=2))
        else:
            for x1, y1, x2, y2, score, cls in dets:
                label = pipe.detector.classes[cls] if cls < len(pipe.detector.classes) else cls
                print(f"{label:<7} score={score:.3f}  box=({x1},{y1})-({x2},{y2})")
        return

    result = pipe.predict(args.image)

    if args.json:
        print(json.dumps(result.to_dict(), ensure_ascii=False, indent=2))
    else:
        print(f"文字坐标（点击顺序）：{result.boxes}")
        print(f"耗时：{result.timings_ms.get('total', 0):.0f}ms")
        if result.error:
            print(f"[error] {result.error}")
        if result.warning:
            print(f"[warn] {result.warning}")

    if args.show and result.boxes:
        rgb = load_rgb(args.image)
        labels = [str(i + 1) for i in range(len(result.boxes))]
        out = save_annotated(rgb, result.boxes, args.show, labels=labels, points=result.points)
        print(f"[show] {out}")


if __name__ == "__main__":
    main()
