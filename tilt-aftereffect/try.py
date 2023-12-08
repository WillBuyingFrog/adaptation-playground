import numpy as np
import os
from PIL import Image

# 定义图像宽高
IMAGE_WIDTH = 416
IMAGE_HEIGHT = 416

# 定义初始阈值
INITIAL_THRESHOLD = 10

# 用于存储图像的列表
image_list = []

# 定义单个比较区域大小
h, w = 5, 5

# 用于存储图像每个区域变化阈值的numpy array，初始值为INITIAL_THRESHOLD
image_threshold = np.full((IMAGE_HEIGHT - h, IMAGE_WIDTH - w), INITIAL_THRESHOLD)

def get_average_image(fr, to):
    # 获取image_list[fr:to]的平均值
    return np.average(image_list[fr:to], axis=0)


if __name__ == '__main__':

    # 文件夹路径
    # 路径是这里固定的，具体是用os.join得出当前文件夹路径下的images文件夹
    folder_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'images')

    # 获取文件夹中的所有文件
    file_list = os.listdir(folder_path)


    # 图像个数计数器
    image_count = 0

    # 遍历文件夹中的每个文件
    for file_name in file_list:

        image_count += 1

        # 构建图像文件的完整路径
        file_path = os.path.join(folder_path, file_name)
        
        # 使用PIL库打开图像文件
        image = Image.open(file_path)

        # 强制转换图像大小
        image = image.resize((IMAGE_WIDTH, IMAGE_HEIGHT))
        
        # 将图像转换为numpy数组并添加到列表中
        image_array = np.array(image)
        image_list.append(image_array)

    # 将图像列表转换为numpy数组
    image_array = np.array(image_list)

    # 初始化所有区域的阈值，初始阈值为
    

    # 定义区域
    regions = []
    for y in range(0, IMAGE_HEIGHT - h):
        for x in range(0, IMAGE_WIDTH - w):
            regions.append((x, y, w, h))

    for i in range(1, image_count):

        current_average = get_average_image(max(i-10, 0), i)

        diffs = np.zeros((IMAGE_HEIGHT - h, IMAGE_WIDTH - w))

        for _ in range(len(regions)):
            xx, yy, ww, hh = regions[_]
            diffs[yy, xx] = np.sum(np.abs(image_array[i, yy:yy+hh, xx:xx+ww] - current_average[yy:yy+hh, xx:xx+ww])) / (xx * yy)

        # 得出diffs中所有比阈值大的数值的索引
        indexes = np.where(diffs > image_threshold)

        print(indexes)

        print(indexes[0].shape)