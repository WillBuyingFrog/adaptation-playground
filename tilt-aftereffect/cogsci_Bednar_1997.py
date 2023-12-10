import os
import argparse
import torch
import numpy as np

import frogtools

# --------
# 图像参数
# --------

# 定义输入图像的宽高
# 原始论文输入图像宽高为192，这里扩大2.75倍
IMAGE_HEIGHT = 528
IMAGE_WIDTH = 528

# 输入的图像
images = []

# ---------
# 模型参数
# ---------

# 定义皮层模型的神经元个数，神经元为2维排布
# 原始论文的神经元个数为24*24，这里扩大2.75倍
CORTIAL_WIDTH = 66
CORTIAL_HEIGHT = 66

# 每个神经元在图像上的感受野宽高
RECEPTIVE_FIELD_WIDTH = 8
RECEPTIVE_FIELD_HEIGHT = 8

# 从图像（视网膜神经节细胞，RGC）到皮层可看作一个连接层，感受野大小为前面定义的感受野宽高相乘
# 此处定义的是图像到皮层的连接层的权重，权重应该为一个PyTorch里的向量，其大小应该为 神经元层宽 * 神经元层高 * 感受野宽 * 感受野高
# 初始值设置为0到1之间的随机float32
afferent_weight = torch.rand(CORTIAL_WIDTH * CORTIAL_HEIGHT * RECEPTIVE_FIELD_WIDTH * RECEPTIVE_FIELD_HEIGHT).float()


# 定义兴奋侧向连接权重
exitatory_lateral_weight = torch.rand(CORTIAL_WIDTH * CORTIAL_HEIGHT * CORTIAL_WIDTH * CORTIAL_HEIGHT).float()

# 定义抑制侧向连接权重
intibiory_lateral_weight = torch.rand(CORTIAL_WIDTH * CORTIAL_HEIGHT * CORTIAL_WIDTH * CORTIAL_HEIGHT).float()



if __name__ == "__main__":
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--type', type=str, default='video', help="Input images source. 'video' for video, 'folder' for images in the 'images' subfolder.")
    parser.add_argument('--video_name', type=str, default='video.mp4', help='video name')

    args = parser.parse_args()

    if args.type == 'video':
        images = frogtools.get_images_from_video(args.video_name, IMAGE_WIDTH, IMAGE_HEIGHT, output_format='torch')
    elif args.type == 'folder':
        images = frogtools.get_images_from_folder('images', IMAGE_WIDTH, IMAGE_HEIGHT, output_format='torch')
    
    
    # 遍历所有图片
    for _ in range(len(images)):
        current_image = images[_]

        # 计算当前图片输入下，皮层神经元的输出（不含侧向连接影响）
        # 计算公式为：神经元输出 = 图像中该神经元对应的感受野 * 到皮层连接权重
        # 神经元在行或是列上的个数和图像宽高，分别都对应8倍关系，所以第i行第j列的神经元应该对应图像中(i-1)*8+1到i*8行，(j-1)*8+1到j*8列的像素
        