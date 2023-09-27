# !/usr/bin/env python
# -*-coding:utf-8 -*-

"""
# File       : zz_onnx.py
# Time       ：2022/10/21 17:28
# Author     ：yujia
# version    ：python 3.6
# Description：
"""
import cv2
import numpy as np
import onnxruntime
import time
import random
from io import BytesIO
from PIL import Image

from .orientation import non_max_suppression, tag_images

np.set_printoptions(precision=4)


class YOLOV5_ONNX(object):
    def __init__(self,onnx_path, classes, providers=None):
        '''初始化onnx'''
        if not providers:
            providers = ['CPUExecutionProvider']
        self.onnx_session=onnxruntime.InferenceSession(onnx_path, providers=providers)
        # print(onnxruntime.get_device())
        self.input_name=self.get_input_name()
        self.output_name=self.get_output_name()
        self.classes=classes
        self.img_size = 640

    def get_input_name(self):
        '''获取输入节点名称'''
        input_name=[]
        for node in self.onnx_session.get_inputs():
            input_name.append(node.name)

        return input_name

    def get_output_name(self):
        '''获取输出节点名称'''
        output_name=[]
        for node in self.onnx_session.get_outputs():
            output_name.append(node.name)

        return output_name

    def get_input_feed(self,image_tensor):
        '''获取输入tensor'''
        input_feed={}
        for name in self.input_name:
            input_feed[name]=image_tensor

        return input_feed

    def letterbox(self,img, new_shape=(640, 640), color=(114, 114, 114), auto=False, scaleFill=False, scaleup=True,
                  stride=32):
        '''图片归一化'''
        # Resize and pad image while meeting stride-multiple constraints
        shape = img.shape[:2]  # current shape [height, width]
        if isinstance(new_shape, int):
            new_shape = (new_shape, new_shape)

        # Scale ratio (new / old)
        r = min(new_shape[0] / shape[0], new_shape[1] / shape[1])
        if not scaleup:  # only scale down, do not scale up (for better test mAP)
            r = min(r, 1.0)

        # Compute padding
        ratio = r, r  # width, height ratios

        new_unpad = int(round(shape[1] * r)), int(round(shape[0] * r))
        dw, dh = new_shape[1] - new_unpad[0], new_shape[0] - new_unpad[1]  # wh padding

        if auto:  # minimum rectangle
            dw, dh = np.mod(dw, stride), np.mod(dh, stride)  # wh padding
        elif scaleFill:  # stretch
            dw, dh = 0.0, 0.0
            new_unpad = (new_shape[1], new_shape[0])
            ratio = new_shape[1] / shape[1], new_shape[0] / shape[0]  # width, height ratios

        dw /= 2  # divide padding into 2 sides
        dh /= 2

        if shape[::-1] != new_unpad:  # resize
            img = cv2.resize(img, new_unpad, interpolation=cv2.INTER_LINEAR)

        top, bottom = int(round(dh - 0.1)), int(round(dh + 0.1))
        left, right = int(round(dw - 0.1)), int(round(dw + 0.1))

        img = cv2.copyMakeBorder(img, top, bottom, left, right, cv2.BORDER_CONSTANT, value=color)  # add border
        return img, ratio, (dw, dh)

    def clip_coords(self,boxes, img_shape):
        '''查看是否越界'''
        # Clip bounding xyxy bounding boxes to image shape (height, width)
        boxes[:, 0].clip(0, img_shape[1])  # x1
        boxes[:, 1].clip(0, img_shape[0])  # y1
        boxes[:, 2].clip(0, img_shape[1])  # x2
        boxes[:, 3].clip(0, img_shape[0])  # y2

    def scale_coords(self,img1_shape, coords, img0_shape, ratio_pad=None):
        '''
        坐标对应到原始图像上，反操作：减去pad，除以最小缩放比例
        :param img1_shape: 输入尺寸
        :param coords: 输入坐标
        :param img0_shape: 映射的尺寸
        :param ratio_pad:
        :return:
        '''

        # Rescale coords (xyxy) from img1_shape to img0_shape
        if ratio_pad is None:  # calculate from img0_shape
            gain = min(img1_shape[0] / img0_shape[0], img1_shape[1] / img0_shape[1])  # gain  = old / new,计算缩放比率
            pad = (img1_shape[1] - img0_shape[1] * gain) / 2, (
                        img1_shape[0] - img0_shape[0] * gain) / 2  # wh padding ，计算扩充的尺寸
        else:
            gain = ratio_pad[0][0]
            pad = ratio_pad[1]

        coords[:, [0, 2]] -= pad[0]  # x padding，减去x方向上的扩充
        coords[:, [1, 3]] -= pad[1]  # y padding，减去y方向上的扩充
        coords[:, :4] /= gain  # 将box坐标对应到原始图像上
        self.clip_coords(coords, img0_shape)  # 边界检查
        return coords

    def to_numpy(self, img, shape):
        # 超参数设置
        img_size = shape
        # img_size = (640, 640)  # 图片缩放大小
        # 读取图片
        # src_img = cv2.imread(img_path)
        src_img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
        start = time.time()
        src_size = src_img.shape[:2]

        # 图片填充并归一化
        img = self.letterbox(src_img, img_size, stride=32)[0]
        # Convert
        img = img[:, :, ::-1].transpose(2, 0, 1)  # BGR to RGB, to 3x416x416
        img = np.ascontiguousarray(img)

        # 归一化
        img = img.astype(dtype=np.float32)
        img /= 255.0

        # # BGR to RGB
        # img = img[:, :, ::-1].transpose(2, 0, 1)
        # img = np.ascontiguousarray(img)

        # 维度扩张
        img = np.expand_dims(img, axis=0)
        # print('img resuming: ', time.time() - start)
        # 前向推理
        # start=time.time()

        return img


    def infer(self,img_path):
        '''执行前向操作预测输出'''
        # 超参数设置
        img_size=(640,640) #图片缩放大小
        # 读取图片
        src_img=cv2.imread(img_path)
        start=time.time()
        src_size=src_img.shape[:2]

        # 图片填充并归一化
        img=self.letterbox(src_img,img_size,stride=32)[0]

        # Convert
        img = img[:, :, ::-1].transpose(2, 0, 1)  # BGR to RGB, to 3x416x416
        img = np.ascontiguousarray(img)


        # 归一化
        img=img.astype(dtype=np.float32)
        img/=255.0

        # # BGR to RGB
        # img = img[:, :, ::-1].transpose(2, 0, 1)
        # img = np.ascontiguousarray(img)

        # 维度扩张
        img=np.expand_dims(img,axis=0)
        # print('img resuming: ',time.time()-start)
        # 前向推理
        # start=time.time()
        input_feed=self.get_input_feed(img)
        # ort_inputs = {self.onnx_session.get_inputs()[0].name: input_feed[None].numpy()}
        pred = self.onnx_session.run(None, input_feed)[0]
        results = non_max_suppression(pred, 0.5,0.5)
        # print('onnx resuming: ',time.time()-start)
        # pred=self.onnx_session.run(output_names=self.output_name,input_feed=input_feed)
        # print(results)
        "[tensor([[104.22188, 221.40257, 537.10876, 490.15454,   0.79675,   0.00000]])]"


        #映射到原始图像
        img_shape=img.shape[2:]
        # print(img_size)
        if results:
            for det in results:  # detections per image
                if det is not None and len(det):
                    det[:, :4] = self.scale_coords(img_shape, det[:, :4],src_size).round()
            # print(time.time()-start)
            if det is not None and len(det):
                self.draw(src_img, det)

    def plot_one_box(self,x, img, color=None, label=None, line_thickness=None):
        # Plots one bounding box on image img
        tl = line_thickness or round(0.002 * (img.shape[0] + img.shape[1]) / 2) + 1  # line/font thickness
        color = color or [random.randint(0, 255) for _ in range(3)]
        c1, c2 = (int(x[0]), int(x[1])), (int(x[2]), int(x[3]))
        cv2.rectangle(img, c1, c2, color, thickness=tl, lineType=cv2.LINE_AA)
        if label:
            tf = max(tl - 1, 1)  # font thickness
            t_size = cv2.getTextSize(label, 0, fontScale=tl / 3, thickness=tf)[0]
            c2 = c1[0] + t_size[0], c1[1] - t_size[1] - 3
            cv2.rectangle(img, c1, c2, color, -1, cv2.LINE_AA)  # filled
            cv2.putText(img, label, (c1[0], c1[1] - 2), 0, tl / 3, [225, 255, 255], thickness=tf, lineType=cv2.LINE_AA)

    def draw(self,img, boxinfo):
        colors = [[random.randint(0, 255) for _ in range(3)] for _ in range(len(self.classes))]
        for *xyxy, conf, cls in boxinfo:
            label = '%s %.2f' % (self.classes[int(cls)], conf)
            # print('xyxy: ', xyxy)
            self.plot_one_box(xyxy, img, label=label, color=colors[int(cls)], line_thickness=1)

        # cv2.namedWindow("dst",0)
        # cv2.imshow("dst", img)
        cv2.imwrite("res1.jpg",img)
        # cv2.waitKey(0)
        # cv2.imencode('.jpg', img)[1].tofile(os.path.join(dst, id + ".jpg"))
        return 0

    def decect(self, file):
        # 图片转换为矩阵
        if isinstance(file, np.ndarray):
            img = Image.fromarray(file)
        elif isinstance(file, bytes):
            img = Image.open(BytesIO(file))
        else:
            img = Image.open(file)
        img = img.convert('RGB')
        img = np.array(img)
        image_numpy = self.to_numpy(img, shape=(self.img_size, self.img_size))
        input_feed = self.get_input_feed(image_numpy)
        pred = self.onnx_session.run(None, input_feed)[0]
        pred = non_max_suppression(pred, 0.5, 0.5)
        res = tag_images(img, pred, self.img_size, self.classes, 0.5)
        return res


if __name__=="__main__":
    # onnx_path = r"D:\CodeFiles\yolov5\runs\train\exp6\weights\best.onnx"
    # img_path = r"D:\CodeFiles\idcard\4cde7c674172638c97e9c2bcdb3e770.jpg"
    onnx_path = r"D:\CodeFiles\yolov5\runs\train\exp3\weights\best.onnx"
    img_path = r"111.png"
    classes = ['a', 'b']
    model=YOLOV5_ONNX(onnx_path=onnx_path, classes=classes)
    model.infer(img_path=img_path)
    s = model.decect(img_path)
    print(s)
    "[array([[104.2219, 221.4026, 537.1088, 490.1545,   0.7967,   0.    ]])]"
    "[tensor([[104.22188, 221.40257, 537.10876, 490.15454,   0.79675,   0.00000]])]"

