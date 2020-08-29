#!/usr/bin/env python

# -*- coding: utf-8 -*-

"""
@author: jiajia
@file: orientation.py
@time: 2020/8/17 16:00
"""
#!/usr/bin/env python

# -*- coding: utf-8 -*-

"""
@author: jiajia
@file: demo.py
@time: 2020/8/13 13:44
"""
"""
框选识别模块
"""
import time
from torch.autograd import Variable
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.ticker import NullLocator

from src.utils.models import *
from src.utils.datasets import *
from src.utils.utils import *
from src.setting import yolo_opt as opt


"""加载模型"""
if opt.GPU:
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
else:
    device = torch.device("cpu")
# Set up model
model = Darknet(opt.model_def, img_size=opt.img_size).to(device)
if opt.weights_path.endswith(".weights"):
    # Load darknet weights
    model.load_darknet_weights(opt.weights_path)
else:
    # Load checkpoint weights
    # model.load_state_dict(torch.load(opt.weights_path))
    model.load_state_dict(
        torch.load(opt.weights_path, map_location="cuda:0" if torch.cuda.is_available() and opt.GPU else "cpu"))

model.eval()  # Set in evaluation mode

classes = load_classes(opt.class_path)  # Extracts class labels from file
Tensor = torch.cuda.FloatTensor if torch.cuda.is_available() and opt.GPU else torch.FloatTensor
"""加载模型"""


def tag_images(imgs, img_detections):
    """图片展示"""
    # Bounding-box colors
    results = []
    cmap = plt.get_cmap("tab20b")
    colors = [cmap(i) for i in np.linspace(0, 1, 20)]


    # Iterate through images and save plot of detections
    for img_i, (path, detections) in enumerate(zip(imgs, img_detections)):


        # Create plot
        img = np.array(Image.open(path))
        plt.figure()
        fig, ax = plt.subplots(1)
        ax.imshow(img)

        # Draw bounding boxes and labels of detections
        if detections is not None:
            # Rescale boxes to original image
            detections = rescale_boxes(detections, opt.img_size, img.shape[:2])
            unique_labels = detections[:, -1].cpu().unique()
            n_cls_preds = len(unique_labels)
            bbox_colors = random.sample(colors, n_cls_preds)
            for x1, y1, x2, y2, conf, cls_conf, cls_pred in detections:
                # print("\t+ Label: %s, Conf: %.5f" % (classes[int(cls_pred)], cls_conf.item()))

                box_w = x2 - x1
                box_h = y2 - y1

                color = bbox_colors[int(np.where(unique_labels == int(cls_pred))[0])]
                # Create a Rectangle patch
                bbox = patches.Rectangle((x1, y1), box_w, box_h, linewidth=2, edgecolor=color, facecolor="none")
                # print('坐标位置为：', [int(i) for i in (x1, y1, box_w, box_h)])
                results.append(
                    {
                        "crop": [int(i) for i in (x1, y1, x2, y2)],
                        "classes": classes[int(cls_pred)]

                    }
                )
                # Add the bbox to the plot
                ax.add_patch(bbox)
                # Add label
                plt.text(
                    x1,
                    y1,
                    s=classes[int(cls_pred)],
                    color="white",
                    verticalalignment="top",
                    bbox={"color": color, "pad": 0},
                )
        else:
            print("识别失败")
        # Save generated image with detections
        plt.axis("off")
        plt.gca().xaxis.set_major_locator(NullLocator())
        plt.gca().yaxis.set_major_locator(NullLocator())
        filename = path.split("/")[-1].split(".")[0]
        # plt.show()
        # plt.savefig(f"{output_folder}/{filename}.png", bbox_inches="tight", pad_inches=0.0)
        plt.close()
    return results


def open_picture(img_path):
    """打开图片"""
    img = transforms.ToTensor()(Image.open(img_path).convert('RGB'))
    # Pad to square resolution
    img, _ = pad_to_square(img, 0)
    # Resize
    img = resize(img, opt.img_size)

    # 扩充维度
    img = torch.unsqueeze(img, 0)
    return img_path, img


def location_predict(img_path):
    """位置预测"""
    imgs = []  # Stores image paths
    img_detections = []  # Stores detections for each image index

    img_paths, input_imgs = open_picture(img_path)
    # print(img_path, input_imgs)

    # Get detections
    prev_time = time.time()
    with torch.no_grad():
        input_imgs = Variable(input_imgs.type(Tensor))
        detections = model(input_imgs)
        detections = non_max_suppression(detections, opt.conf_thres, opt.nms_thres)
    # print("预测耗时：", time.time() - prev_time)

    # Save image and detections
    imgs.append(img_paths)
    img_detections.extend(detections)
    # print(detections)
    res = tag_images(imgs, img_detections)
    return res


if __name__ == "__main__":
    path = "../test/2.jpg"
    res, input_imgs = location_predict(path)


    # path = "data/captcha/test/captcha_4497.png"
    # while True:
    #     path = input("请输入图片路径：")
    #     predict(path)





