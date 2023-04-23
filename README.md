# 点击选择文字验证码识别
文字点选、选字、选择文字验证码识别  
**特性**  
识别速度约在100~300ms之间  
高达97%的准确率  
小样本训练（此模型训练采用了300张验证码）  
windows下python3.6、python3.8、python3.10测试使用通过  
低消耗，代码经编译后在低配置机器上也可运行  

## 效果演示
![Image text](./docs/res.gif)  

```angular2html
项目结构
---app 服务相关代码
---model 模型文件
---src  项目相关代码
```
## 更新说明
#### 20230423更新  
更改检测识别模型，修改返回结构

## 免责声明
**本项目仅供学习交流使用，请勿用于非法用途，不得在任何商业使用，本人不承担任何法律责任。**


## 环境准备 
3、pip install -r requirements.txt
## 如何使用
``` bash
#普通使用
python dome.py
服务启动方式
python service.py
启动后访问http://127.0.0.1:8000/docs#/查看接口文档
bilbil演示
python bilbil.py
```  
结果如下  
```json
[
  [123, 174, 190, 243],
  [222, 188, 295, 265],
  [84, 72, 158, 138],
  [32, 197, 107, 279]
]
```


| 原始 | 检测 | 识别 |
| --- | --- |--- |
| <img src="./docs/res.jpg" width="300" height="300"> | <img src="./docs/res1.jpg" width="300" height="300"> | <img src="./docs/res2.jpg" width="300" height="300">|

 




# 请作者者喝可乐**o(*￣︶￣*)o**  

| Wechat Pay | Ali Pay |
| --- | --- |
| <img src="./docs/wechat.jpg" height="300" /> | <img src="./docs/Ali.png" height="300" /> |

## 有什么问题或需求欢迎各位在lssues中提问或联系邮件**yj970814@163.com**
### 点个**star**再走呗！ 


