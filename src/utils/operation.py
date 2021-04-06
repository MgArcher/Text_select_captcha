"""
@author: jiajia
@file: operation.py
@time: 2021/3/28 15:31
"""
from io import BytesIO
import onnxruntime
import numpy as np
from PIL import Image


class ONNXModel(object):
    def __init__(self, onnx_path):
        """
        :param onnx_path:
        """
        self.onnx_session = onnxruntime.InferenceSession(onnx_path)
        self.input_name = self.get_input_name(self.onnx_session)
        self.output_name = self.get_output_name(self.onnx_session)
        # print("input_name:{}".format(self.input_name))
        # print("output_name:{}".format(self.output_name))

    def get_output_name(self, onnx_session):
        """
        output_name = onnx_session.get_outputs()[0].name
        :param onnx_session:
        :return:
        """
        output_name = []
        for node in onnx_session.get_outputs():
            output_name.append(node.name)
        return output_name

    def get_input_name(self, onnx_session):
        """
        input_name = onnx_session.get_inputs()[0].name
        :param onnx_session:
        :return:
        """
        input_name = []
        for node in onnx_session.get_inputs():
            input_name.append(node.name)
        return input_name

    def get_input_feed(self, input_name, image_numpy):
        """
        input_feed={self.input_name: image_numpy}
        :param input_name:
        :param image_numpy:
        :return:
        """
        input_feed = {}
        for name in input_name:
            input_feed[name] = image_numpy
        return input_feed

    def to_numpy(self, file, shape, gray=False):
        if isinstance(file, np.ndarray):
            img = Image.fromarray(file)
        elif isinstance(file, bytes):
            img = Image.open(BytesIO(file))
            pass
        else:
            img = Image.open(file)

        widht, hight = shape
         # 改变大小 并保证其不失真
        img = img.convert('RGB')
        if gray:
            img = img.convert('L')
        img = img.resize((widht, hight), Image.ANTIALIAS)

        # 转换成矩阵
        image_numpy = np.array(img) # (widht, hight, 3)
        if gray:
            image_numpy = np.expand_dims(image_numpy,0)
            image_numpy = image_numpy.transpose(0, 1, 2)
        else:
            image_numpy = image_numpy.transpose(2,0,1) # 转置 (3, widht, hight)
        image_numpy = np.expand_dims(image_numpy,0)
        # 数据归一化
        image_numpy = image_numpy.astype(np.float32) / 255.0
        return image_numpy


class CNN(ONNXModel):
    def __init__(self, onnx_path="cnn.onnx", char_dict="ch_sim_char_7255.txt"):
        super(CNN, self).__init__(onnx_path)
        with open(char_dict, 'r', encoding='utf-8') as f:
            data = f.read()
            self.characters = data.split('\n')

    def decect(self, img):
        # 图片转换为矩阵
        image_numpy = self.to_numpy(img, shape=(64,64))

        input_feed = self.get_input_feed(self.input_name, image_numpy)
        out = self.onnx_session.run(self.output_name, input_feed=input_feed)
        pic = int(np.argmax(out[0]))
        res = self.characters[pic]

        return res

    def batch_decect(self, image_numpy):
        input_feed = self.get_input_feed(self.input_name, image_numpy)
        out = self.onnx_session.run(self.output_name, input_feed=input_feed)[0]
        return out


class CRNN(ONNXModel):
    def __init__(self, onnx_path="crnn.onnx", char_dict="ch_sim_char_7255.txt"):
        super(CRNN, self).__init__(onnx_path)
        with open(char_dict, 'r', encoding='utf-8') as f:
            data = f.read()
            self.characters = data.split('\n')

    def strLabelConverter(self, res, alphabet):
        N = len(res)
        raw = []
        for i in range(N):
            if res[i] != 0 and (not (i > 0 and res[i - 1] == res[i])):
                raw.append(alphabet[res[i] - 1])
        return ''.join(raw)

    def decect(self, img):
        # 图片转换为矩阵
        image_numpy = self.to_numpy(img, shape=(100, 32), gray=True)
        input_feed = self.get_input_feed(self.input_name, image_numpy)
        out = self.onnx_session.run(self.output_name, input_feed=input_feed)[0]
        pic = np.argmax(out, axis=2)
        pic = pic.T
        res = self.strLabelConverter(pic[0], alphabet=self.characters)
        return res





import torch
from src.utils.orientation import non_max_suppression, tag_images

class YOLO(ONNXModel):
    def __init__(self, onnx_path="crnn.onnx"):
        super(YOLO, self).__init__(onnx_path)
        # 训练所采用的输入图片大小
        self.img_size = 640
        self.img_size_h = self.img_size_w = self.img_size
        self.batch_size = 1

        self.num_classes = 2
        self.anchors = [[10, 13, 16, 30, 33, 23], [30, 61, 62, 45, 59, 119], [116, 90, 156, 198, 373, 326]]
        self.classes = ['target', 'title']

    def to_numpy(self, file, shape, gray=False):

        def letterbox_image(image, size):
            iw, ih = image.size
            w, h = size
            scale = min(w / iw, h / ih)
            nw = int(iw * scale)
            nh = int(ih * scale)

            image = image.resize((nw, nh), Image.BICUBIC)
            new_image = Image.new('RGB', size, (128, 128, 128))
            new_image.paste(image, ((w - nw) // 2, (h - nh) // 2))
            return new_image

        if isinstance(file, np.ndarray):
            img = Image.fromarray(file)
        elif isinstance(file, bytes):
            img = Image.open(BytesIO(file))
            pass
        else:
            img = Image.open(file)

        resized = letterbox_image(img, (self.img_size_w, self.img_size_h))
        img_in = np.transpose(resized, (2, 0, 1)).astype(np.float32)  # HWC -> CHW
        img_in = np.expand_dims(img_in, axis=0)
        img_in /= 255.0
        return img_in

    def detect(self, outputs):
        """编辑实现yolov5最后的预测层(有点问题，最后输出的图片与yolo未转换的相比有差别)"""
        boxs = []
        a = torch.tensor(self.anchors).float().view(3, -1, 2)
        anchor_grid = a.clone().view(3, 1, -1, 1, 1, 2)
        if len(outputs) == 4:
            outputs = [outputs[1], outputs[2], outputs[3]]
        for index, out in enumerate(outputs):
            out = torch.from_numpy(out)
            batch = out.shape[1]
            feature_w = out.shape[2]
            feature_h = out.shape[3]

            # Feature map corresponds to the original image zoom factor
            stride_w = int(self.img_size_w / feature_w)
            stride_h = int(self.img_size_h / feature_h)

            grid_x, grid_y = np.meshgrid(np.arange(feature_w), np.arange(feature_h))

            # cx, cy, w, h
            pred_boxes = torch.FloatTensor(out[..., :4].shape)
            pred_boxes[..., 0] = (torch.sigmoid(out[..., 0]) * 2.0 - 0.5 + grid_x) * stride_w  # cx
            pred_boxes[..., 1] = (torch.sigmoid(out[..., 1]) * 2.0 - 0.5 + grid_y) * stride_h  # cy
            pred_boxes[..., 2:4] = (torch.sigmoid(out[..., 2:4]) * 2) ** 2 * anchor_grid[index]  # wh

            conf = torch.sigmoid(out[..., 4])
            pred_cls = torch.sigmoid(out[..., 5:])

            output = torch.cat((pred_boxes.view(self.batch_size, -1, 4),
                                conf.view(self.batch_size, -1, 1),
                                pred_cls.view(self.batch_size, -1, self.num_classes)),
                               -1)
            boxs.append(output)

        outputx = torch.cat(boxs, 1)
        return outputx

    def decect(self, img):
        # 图片转换为矩阵
        image_numpy = self.to_numpy(img, shape=(self.img_size, self.img_size))
        input_feed = self.get_input_feed(self.input_name, image_numpy)
        outputs = self.onnx_session.run(self.output_name, input_feed=input_feed)
        outputx = self.detect(outputs)
        pred = non_max_suppression(outputx)
        res = tag_images(img, pred, self.img_size, self.classes)
        return res


if __name__ == '__main__':

    yolo = YOLO("C:\CodeFiles\image\wordChoice\Text_select_captcha/src/model/yolo.onnx")
    img = r"C:\CodeFiles\image\wordChoice\Text_select_captcha\img\1234.jpg"
    pred = yolo.decect(img)

    data = []
    for i in pred:
        i['content'] = ''
        data.append(i)
    from drawing import draw
    draw(img, data)
