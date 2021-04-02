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


if __name__ == '__main__':
    import time
    cnn_model_path= "../model/cnn.onnx"
    file = "../../img/1009.jpg"
    rnet1 = CNN(cnn_model_path)
    s = time.time()
    out = rnet1.decect(file)
    print(out)
    print(time.time() -s)

    X = np.zeros((2, 3, 64, 64), dtype=np.float32)
    X[0] = rnet1.to_numpy(file, shape=(64,64))
    X[1] = rnet1.to_numpy(file, shape=(64,64))
    out = rnet1.batch_decect(X)
    # crnn_model_path="crnn.onnx"
    # file = "9653.jpg"
    # rnet2 = CRNN(crnn_model_path)
    # s = time.time()
    # out = rnet2.decect(file)
    # print(out)
    # print(time.time() -s)
