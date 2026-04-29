好的，这是为您润色优化后的项目文档。主要改进了语言表达、逻辑结构和格式规范，同时保留了所有关键信息、链接和图片。

---

# 点击选择文字验证码识别

**文字点选、选字、选择文字验证码识别**

## 项目特点

- **识别速度**：约 300~500 毫秒
- **准确率**：96%
- **训练样本**：小样本学习，仅需约 300 张验证码图片
- **资源占用**：低配置机器（1 核 2G 服务器）可无压力运行
- **平台支持**：全平台支持，python3.8+以上版本
- **可扩展性**：支持自行训练模型，不受固定平台限制

## 效果演示

![演示动图](./docs/res.gif)

## 免责声明

**本项目旨在研究深度学习在验证码攻防上的应用。仅供学习交流使用，请勿用于非法用途，禁止任何商业使用。作者不承担任何法律责任。**

## 道满PythonAI

[「道满PythonAI」](https://www.daomanpy.com/) - 2026 领先的中文 Python 与 AI 体系化教程。涵盖 Python 基础、Web 后端开发、高阶爬虫、JS 逆向、计算机视觉、NLP 及大模型 (LLM) 实战，打通从零基础到 AI 全栈工程师的职业路径。

![道满PythonAI Logo](./docs/46354c4191deebbf378736cfc35cc749.png)

## 支持作者

### 请作者喝可乐 **o(*￣︶￣*)o**

| 微信支付 | 支付宝 |
| --- | --- |
| <img src="./docs/wechat.jpg" height="500" /> | <img src="./docs/Ali.png" height="500" /> |

### 定制需求或问题反馈

- 邮件：**yj970814@163.com**
- 欢迎在 Issues 中提问或联系邮箱

## 快速开始

### 1. 环境准备

```bash
pip install -r requirements.txt
```

### 2. 普通使用示例

```bash
python demo.py
```

### 3. 启动服务（API 方式）

```bash
python service.py
```
启动后访问 `http://127.0.0.1:8000/docs#/` 查看接口文档。

### 4. Bilibili 自动化演示

```bash
python bilbil.py
```

### Python 代码调用示例

```python
from src.captcha import TextSelectCaptcha

"""
参数说明：
per_path: 孪生网络模型文件路径
yolo_path: 目标检测模型文件路径
sign: True 表示使用加密后的 .bin 模型文件，False 表示使用 .onnx 文件
"""
cap = TextSelectCaptcha()

image_path = "docs/res.jpg"
result = cap.run(image_path)


result = cap.run_dict(image_path)
```

#### 返回结果示例

```
文字坐标： [[119, 174, 193, 244], [223, 189, 298, 267], [87, 69, 158, 140], [31, 196, 109, 275]] 耗时：163ms
```
```
调用run_dict 返回字典形式数据
result = cap.run_dict(image_path)
{
    "imgW": 344,
    "imgH": 384,
    "point": [
        {
            "x_rel": 159.0,
            "y_rel": 209.0
        },
        {
            "x_rel": 260.5,
            "y_rel": 229.5
        },
        {
            "x_rel": 122.5,
            "y_rel": 105.0
        },
        {
            "x_rel": 69.5,
            "y_rel": 235.5
        }
    ],
    "corp": [
        {
            "x1": 126,
            "y1": 174,
            "x2": 192,
            "y2": 244
        },
        {
            "x1": 225,
            "y1": 191,
            "x2": 296,
            "y2": 268
        },
        {
            "x1": 86,
            "y1": 70,
            "x2": 159,
            "y2": 140
        },
        {
            "x1": 32,
            "y1": 196,
            "x2": 107,
            "y2": 275
        }
    ]
}
```

> 返回的坐标顺序即为文字需要点击的顺序。

| 原始图片 | 检测结果 | 识别结果 |
| --- | --- | --- |
| <img src="./docs/res.jpg" width="300" height="300"> | <img src="./docs/res1.jpg" width="300" height="300"> | <img src="./docs/res2.jpg" width="300" height="300"> |

## 更新日志

- **2023.04.23**：更换检测与识别模型，优化返回数据结构  
- **2023.08.18**：取消推理代码编译，仅对模型加载部分进行编译；优化 Web 接口  
- **2023.09.22**：新增消消乐验证码破解功能  
- **2026.04.29**：精简代码，优化识别效果，解除平台限制，提升推理速度，简化代码文件  

### 消消乐验证码示例

![消消乐演示](./docs/xiaoxiaole.gif)

> 消消乐识别原理说明：[掘金文章](https://juejin.cn/post/7282889267030016061)

```bash
python xiaoxiaole.py
```
执行后会自动找出最近可被消除的行或列。

- **2023.09.28**：发布 v2 模型，增强泛化能力，提升推理速度（准确率下降至约 90%）  
- **2023.11.12**：优化识别准确率，略微提升推理速度，重构代码结构  
- **2024.08.17**：修复部分 Bug，解决 `bilbil.py` 代码无法使用及点击位置偏移问题  
- **2024.09.12**：更换孪生网络模型，保持识别速度的同时提升准确率，增加基于 `pyppeteer` 的自动化操作

## 实现流程

### 问题拆解

点选式验证码可以拆解为两个子任务：

#### 1. 检测待点击文字的数量与位置

采用 **YOLOv5** 目标检测模型（本项目使用 `yolov5s6` 预训练模型），通过标注数据集进行训练，实现文字区域的精准定位。

- 标注方式：对背景图中的文字标注为 `char` 类别，对需要点击的文字标注为 `target` 类别。

<div align=center>
<img src="./docs/img.png" width="800" height="400">
<div>YOLO 标注示例</div>
</div>

训练完成后将模型导出为 ONNX 格式，便于 CPU 推理部署。

<div align=center>
<img src="./docs/res1.jpg" width="400" height="400">
<div>YOLO 检测结果示例</div>
</div>

#### 2. 确定点击顺序

采用 **Siamese 孪生网络**，将检测到的文字区域与预定义的字库图片进行匹配，找到最相似的文字，并结合坐标排序规则确定最终点击顺序。

- 训练数据生成：使用已训练好的检测模型从图片中截取目标区域，构造匹配/不匹配的样本对。

<div align=center>
<img src="./docs/img_1.png">
<div>孪生网络样本对示例</div>
</div>

训练完成后同样导出为 ONNX 模型。

<div align=center>
<img src="./docs/myplot1.png" width="320" height="240">
<img src="./docs/myplot2.png" width="320" height="240">
<img src="./docs/myplot3.png" width="320" height="240">
<img src="./docs/myplot4.png" width="320" height="240">
<div>孪生网络输出效果</div>
</div>

### 推理部署

将 YOLO 和 Siamese 模型均转换为 **ONNX** 格式，可在 CPU 上高效运行，降低部署难度，提高运行速度。  
ONNX 支持跨平台、跨框架使用，并可与 C++ / Python 等语言交互。

> **ONNX 简介**：开放神经网络交换格式，由 Facebook 和 Microsoft 联合开发，支持 PyTorch、TensorFlow 等主流框架的模型转换与部署。

### 参考文档

- [YOLOv5 官方仓库](https://github.com/ultralytics/yolov5)
- [Siamese-PyTorch 实现](https://github.com/bubbliiiing/Siamese-pytorch)

### 训练数据集

百度网盘链接：https://pan.baidu.com/s/1IYfxVpanXyqVQ8ZFVOskrg  
提取码：`sp97`

## 给个 Star 再走呗 ⭐️

---

如果您需要针对某个具体环节（如训练、部署、集成）进行进一步优化或解释，请随时告知。