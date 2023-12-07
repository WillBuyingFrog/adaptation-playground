import numpy as np
import os
from PIL import Image


IMAGE_WIDTH = 416
IMAGE_HEIGHT = 416


# 文件夹路径
# 路径是这里固定的，具体是用os.join得出当前文件夹路径下的images文件夹
folder_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'images')

# 获取文件夹中的所有文件
file_list = os.listdir(folder_path)

# 用于存储图像的列表
image_list = []

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

diffs = []

# 定义单个比较区域大小
h, w = 5, 5

# 定义区域
regions = []
for y in range(0, IMAGE_HEIGHT - h):
    for x in range(0, IMAGE_WIDTH - w):
        regions.append((x, y, w, h))

for i in range(1, image_count):
    diff = np.abs(image_array[i][y:y+h, x:x+w] - image_array[i-1][y:y+h, x:x+w])
    diffs.append(diff)
