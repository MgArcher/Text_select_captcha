# 点击选择文字验证码识别
文字点选、选字、选择文字验证码识别  
**特点**  
纯pytorch实现，无需安装其他复杂依赖  
识别速度约在100~200ms之间，使用GPU话会更快  
~~方式二的话速度约为50ms~~  

## 更新说明
本次进行了一次大的版本更新，检测模型从yoloV3->yoloV5,cnn模型和crnn模型都增加了新的数据重新训练了。  
识别速度和准确率都有了大幅度的提升，使用cpu的情况下单次识别速度在100ms以内，识别准确率达到了惊人的99%。  
![Image text](./doc/b9286cf8d1d851f7398cca4f90e5d24.png)   
此次更新把cnn和crnn模型转换成onnx,用来提高cpu推理速度，优化代码结构，修复其中一些bug和删除了一些不必要的代码。  
新版代码克隆可直接使用，旧版代码在dev分支

## 免责声明
**本项目仅供学习交流使用，请勿用于非法用途，不得在任何商业使用，本人不承担任何法律责任。**
## 训练集
  **百度网盘**  
链接：https://pan.baidu.com/s/1FF6A-YZAE1Bofgswp-D29w  
提取码：ceow  
## 实现逻辑 
**识别逻辑**  
方式一  
1、利用yolo框选出给出的文字和图中出现的文字，作为题目  
2、利用crnn识别给定的文字，作为答题范围  
3、根据答题范围，利用cnn预测图片中出现的文字是那个  
![Image text](./doc/fc2b0.png)    
~~方式二~~(kenlm计算的方式总是不太理想，先去掉此种方式)  
1、利用yolo框选图中出现的文字  
2、利用cnn识别图中文字  
3、利用kenlm计算各种组合情况，选择困惑度最低的,获得正确的词语   
![Image text](./doc/xyj.png)   
##  模型文件  
**下载链接**   
http://39.108.219.137/text_select_captcha/model    
  **百度网盘**  
链接：https://pan.baidu.com/s/1FF6A-YZAE1Bofgswp-D29w  
提取码：ceow  
**dev分支**
（下载model文件夹放入到代码所在目录）
模型文件在model目录下  
卷积神经网络模型 cnn_iter.pth（用于识别图片中的文字）  
卷积神经网络+CTCloss模型 ocr-lstm.pth（用于识别标题中的文字）   
yoloV3模型 yolov3_ckpt.pth （用于框选出图片中的文字和标题）    
kenlm统计语言模型 people_chars_lm.klm  （用于计算语序）  
**模型结构**  
模型结构存放在src/utils中

## 环境准备
1、安装python3.6（建议使用anconda）  
2、建立虚拟环境  
3、pip install -r requirements.txt
## 如何使用

``` bash
python dome.py
```  
结果如下  
```json
[
    {
        "crop": [
            231,
            173,
            297,
            248
        ],
        "classes": "target",
        "content": "拌"
    },
    {
        "crop": [
            0,
            344,
            114,
            385
        ],
        "classes": "title",
        "content": "凉拌牛肚"
    },
    {
        "crop": [
            58,
            189,
            125,
            265
        ],
        "classes": "target",
        "content": "牛"
    },
    {
        "crop": [
            231,
            271,
            297,
            343
        ],
        "classes": "target",
        "content": "肚"
    },
    {
        "crop": [
            201,
            79,
            265,
            152
        ],
        "classes": "target",
        "content": "凉"
    }
]
```
![Image text](./doc/123.jpg)  

## 效果演示
**以bilbil登录验证码为例**  
```python bilbil.py```  
![Image text](./doc/bilibili_1.gif)  
![Image text](./doc/bilibili_2.gif)  




# 打赏
如果觉得我的项目对您有用，请随意打赏。您的支持将鼓励我继续创作！  
**o(*￣︶￣*)o**  
![Image text](./doc//20200823220018.png)  
如有什么问题欢迎各位在lssues中提问  
有其他问题或需求请联系邮件**yj970814@163.com**

# 参考资料
https://github.com/ypwhs/captcha_break  
https://github.com/eriklindernoren/PyTorch-YOLOv3  
https://github.com/meijieru/crnn.pytorch  
https://github.com/chineseocr/chineseocr  
https://github.com/JiageWang/hand-writing-recognition

### 点个**star**再走呗！ 


