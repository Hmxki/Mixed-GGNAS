# import os
# from PIL import Image
# import numpy as np
# import cv2
#
#
#
#
# def main():
#     # img_h, img_w = 32, 32
#     img_h, img_w = 224, 224  # 根据自己数据集适当调整，影响不大
#     means, stdevs = [], []
#     img_list = []
#
#     #imgs_path = 'H:\\hmx\\DATA\\idrid\\train\\images'
#     imgs_path = 'H:\\hmx\\code\\paper3\\Fed\\data\\ChestXray14\\images'
#     # imgs_path = '../data/busi/train/images'
#     imgs_path_list = os.listdir(imgs_path)
#     len_ = len(imgs_path_list)
#     i = 0
#     for item in imgs_path_list:
#         img = cv2.imread(os.path.join(imgs_path, item))
#         img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
#         #img = cv2.resize(img, (img_w, img_h))
#
#         img = img[:, :, :, np.newaxis]
#         img_list.append(img)
#         i += 1
#         print(i, '/', len_)
#
#     imgs = np.concatenate(img_list, axis=3)
#     imgs = imgs.astype(np.float32) / 255.
#
#     for i in range(3):
#         pixels = imgs[:, :, i, :].ravel()  # 拉成一行
#         means.append(np.mean(pixels))
#         stdevs.append(np.std(pixels))
#
#     # BGR --> RGB ， CV读取的需要转换，PIL读取的不用转换
#     # means.reverse()
#     # stdevs.reverse()
#
#     print("normMean = {}".format(means))
#     print("normStd = {}".format(stdevs))
#     # img_channels = 3
#     # img_dir = "./CVC/training/images"
#     # #roi_dir = "./CVC/training/masks"
#     # assert os.path.exists(img_dir), f"image dir: '{img_dir}' does not exist."
#     # #assert os.path.exists(roi_dir), f"roi dir: '{roi_dir}' does not exist."
#     #
#     # img_name_list = [i for i in os.listdir(img_dir) if i.endswith(".tif")]
#     # cumulative_mean = np.zeros(img_channels)
#     # cumulative_std = np.zeros(img_channels)
#     # for img_name in img_name_list:
#     #     img_path = os.path.join(img_dir, img_name)
#     #     #ori_path = os.path.join(roi_dir, img_name)
#     #     img = cv2.imread(img_path)
#     #     img = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
#     #     img = np.array(img) / 255.
#     #     #img = np.array(Image.open(img_path)) / 255.
#     #     #roi_img = np.array(Image.open(ori_path).convert('L'))
#     #
#     #     #img = img[roi_img == 255]
#     #     cumulative_mean += img.mean(axis=0)
#     #     cumulative_std += img.std(axis=0)
#     #
#     # mean = cumulative_mean / len(img_name_list)
#     # std = cumulative_std / len(img_name_list)
#     # print(f"mean: {mean}")
#     # print(f"std: {std}")
#
#
# if __name__ == '__main__':
#     main()


import numpy as np
import cv2
import os


# 计算数据集的均值和标准差
from tqdm import tqdm


def compute_mean_std(dataset_path):
    mean = np.zeros(3)
    std = np.zeros(3)
    count = 0

    for filename in tqdm(os.listdir(dataset_path)):

        img_path = os.path.join(dataset_path, filename)
        img = cv2.imread(img_path)  # 以 BGR 方式读取
        img = img.astype(np.float32) / 255.0  # 归一化到 [0,1]

        if img is not None:
            mean += np.mean(img, axis=(0, 1))  # 计算每个通道的均值
            std += np.std(img, axis=(0, 1))  # 计算每个通道的标准差
            count += 1

    mean /= count
    std /= count

    return mean, std


# 数据集路径
dataset_path = 'H:\\hmx\\code\\paper3\\Fed\\data\\ChestXray14\\images'

# 计算均值和标准差
mean_value, std_value = compute_mean_std(dataset_path)

print(f"Mean: {mean_value}")
print(f"Standard Deviation: {std_value}")
