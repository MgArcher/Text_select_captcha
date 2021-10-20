"""
@author: jiajia
@file: operation.py
@time: 2021/3/28 15:31
"""
import numpy as np
from PIL import Image
from io import BytesIO


def open_image(file):
    if isinstance(file, np.ndarray):
        img = Image.fromarray(file)
    elif isinstance(file, bytes):
        img = Image.open(BytesIO(file))
    else:
        img = Image.open(file)
    img = img.convert('RGB')
    img = np.array(img)
    return img


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

    def text_predict(self, res, image_):
        """文本预测"""
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
            elif classes == "target":
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


