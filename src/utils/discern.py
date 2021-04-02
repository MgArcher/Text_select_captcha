"""
@author: jiajia
@file: operation.py
@time: 2021/3/28 15:31
"""
import numpy as np
from PIL import Image
from io import BytesIO


class Text(object):
    def __init__(self, cnn, crnn):
        self.cnn = cnn
        self.crnn = crnn

    def update_text(self, text_list, title):
        res = [i[0] for i in text_list]
        if set(res) == set(title):
            return res

        # 判断是否有重复的文字
        if len(res) != len(set(res)):
            # 判断标题长度与框出来的文字是否长度相同
            if len(text_list) == len(title):
                # 对出现了相同字符的字，把出现概率低的替换
                no_text = set(title) - set([i[0] for i in text_list])
                S = set()
                W = set()
                for i in range(len(text_list)):
                    if text_list[i][0] in S:
                        W.add(text_list[i][0])
                    S.add(text_list[i][0])
                for w in W:
                    qu = [i for i in text_list if i[0] == w]
                    u = max(qu, key=lambda x: x[1])
                    for i in qu:
                        if i != u:
                            number = text_list.index(i)
                            # 随机给一个
                            if no_text:
                                text_list[number] = (no_text.pop())

                results = [i[0] for i in text_list]
            else:
                results = [i[0] for i in text_list]
        else:
            results = res
        return results

    def get_text(self, X, title):
        # 获得每个字的可能概率

        outputs = self.cnn.batch_decect(X)
        text_list = []
        # 获取标签所在的字符位置
        title_Y = [self.cnn.characters.index(i) for i in title]
        # 选取标题中字符中概率最高的
        for i in outputs:
            y = []
            for j in title_Y:
                y.append(i[j])
            # 获得其中的最大值
            if y:
                y = np.mat(y)
                text_list.append((title[np.argmax(y)], y.max()))
            else:
                pic = int(np.argmax(i))
                res = self.cnn.characters[pic]
                text_list.append(res)
        # 修改结果
        results = self.update_text(text_list, title)
        return results

    def rm_repeat_str(self, s):
        plist = list(s)
        b = list(set(plist))
        b.sort(key=plist.index)
        return ''.join(b)

    def open_image(self, file):
        if isinstance(file, np.ndarray):
            img = Image.fromarray(file)
        elif isinstance(file, bytes):
            img = Image.open(BytesIO(file))
        else:
            img = Image.open(file)
        img = np.array(img)
        return img

    def filter_res(self, input, norm=0.9):
        """过滤掉框选错误的结果 文字都是正方形的，过滤掉一些长宽比与标准值相差0.05以上的框"""
        res = []
        m = 0
        n = None
        for inp in input:
            if inp.get("classes") == "target":
                m += 1
            else:
                n = inp.get("content")
        n = len(n) if n else 0
        if n == n:
            return input

        for inp in input:
            if inp.get("classes") == "target":
                crop = inp.get("crop")
                x1, y1, x2, y2 = crop
                ratio = abs(((x1 - x2) / (y1 - y2)) - norm)
                if ratio < 0.04:
                    res.append(inp)
            else:
                res.append(inp)
        return res

    def text_predict(self, res, image_path):
        """文本预测"""

        image_ = self.open_image(image_path)
        zreo = lambda p: [0 if x < 0 else x for x in p]

        # 设置输入cnn形状
        x_len = len([i for i in res if i.get("classes") == "target"])
        X = np.zeros((x_len, 3, 64, 64), dtype=np.float32)
        x_number = 0
        title = ""
        sign = False
        for i, r in enumerate(res):
            classes = r.get("classes")
            crop = r.get("crop")
            crop = zreo(crop)
            x1, y1, x2, y2 = crop
            if classes == "title":
                im = image_[y1:y2, x1:x2]
                title = self.crnn.decect(im)
                # 删除重复出现的字
                # title = "包谷稀饭饭"
                title = self.rm_repeat_str(title)
                res[i]['content'] = title
        res = self.filter_res(res)
        for i, r in enumerate(res):
            classes = r.get("classes")
            crop = r.get("crop")
            crop = zreo(crop)
            x1, y1, x2, y2 = crop
            if classes == "target":
                im = image_[y1:y2, x1:x2]
                X[x_number] = self.cnn.to_numpy(im, shape=(64, 64))
                x_number += 1
                sign = True

        if sign:
            text_list = self.get_text(X, title)
            x_number = 0
            for i, r in enumerate(res):
                classes = r.get("classes")
                if classes == "target":
                    res[i]['content'] = text_list[x_number]
                    x_number += 1

        return res


if __name__ == '__main__':
    path = "../../img/1234.jpg"
    res = [{'crop': [0, 343, 118, 384], 'classes': 'title'}, {'crop': [99, 143, 167, 218], 'classes': 'target'}, {'crop': [252, 261, 320, 337], 'classes': 'target'}, {'crop': [129, 26, 191, 96], 'classes': 'target'}, {'crop': [258, 163, 318, 230], 'classes': 'target'}]
    import time
    from operation import CRNN, CNN
    crnn = CRNN("../model/crnn.onnx")
    cnn = CNN("../model/cnn.onnx")
    text = Text(cnn, crnn)
    s1 = time.time()
    jieguo = text.text_predict(res, path)
    print(jieguo)
    print(time.time() - s1)