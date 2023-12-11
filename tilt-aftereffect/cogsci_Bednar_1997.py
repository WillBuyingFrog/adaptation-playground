import os
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
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


# --------
# 模型实现
# --------


class CortexNetwork(nn.Module):
    def __init__(self, cortial_width, cortial_height, afferent_weights
                 ex_lateral_weight_init, in_lateral_weight_init, gamma_e, gamma_i, alpha_A, alpha_E, alpha_I):
        super(CortexNetwork, self).__init__()
        self.cortial_width = cortial_width
        self.cortial_height = cortial_height
        
        # 前向连接权重的形状是(1, 宽方向神经元数量, 高方向神经元数量, )
        self.afferent_weights = afferent_weights

        self.lateral_weights_excitatory = nn.Parameter(ex_lateral_weight_init)  # 初始化兴奋性侧向权重
        self.lateral_weights_inhibitory = nn.Parameter(in_lateral_weight_init)  # 初始化抑制性侧向权重
        self.gamma_e = gamma_e  # 兴奋性侧向连接的缩放因子
        self.gamma_i = gamma_i  # 抑制性侧向连接的缩放因子
        
        self.alpha_A = alpha_A  # 传入连接的学习率
        self.alpha_E = alpha_E  # 兴奋性连接的学习率
        self.alpha_I = alpha_I  # 抑制性连接的学习率

    
    def forward(self, input, prev_activity):
        # 计算传入激活
        
        
        # 计算侧向激活
        lateral_activation_excitatory = F.linear(prev_activity, self.lateral_weights_excitatory)
        lateral_activation_inhibitory = F.linear(prev_activity, self.lateral_weights_inhibitory)
        
        # 计算总激活
        total_activation = afferent_activation + self.gamma_e * lateral_activation_excitatory + self.gamma_i * lateral_activation_inhibitory
        activity = self.sigmoid_approximation(total_activation)  # 使用逐段线性Sigmoid近似
        
        return activity
    
    def sigmoid_approximation(self, x):
        # 实现逐段线性Sigmoid近似函数
        return torch.clamp(x, min=0)  # 这里只是一个简单的ReLU作为示例
    
    def update_weights(self, presynaptic_activity, postsynaptic_activity):
        # Hebbian权重更新规则
        weight_change_afferent = self.alpha_A * torch.ger(presynaptic_activity, postsynaptic_activity)
        self.afferent_weights.data += weight_change_afferent
        self.afferent_weights.data /= self.afferent_weights.data.sum(0, keepdim=True)  # 归一化
        
        # 更新侧向连接权重
        weight_change_lateral_excitatory = self.alpha_E * torch.ger(postsynaptic_activity, postsynaptic_activity)
        self.lateral_weights_excitatory.data += weight_change_lateral_excitatory
        self.lateral_weights_excitatory.data /= self.lateral_weights_excitatory.data.sum(0, keepdim=True)  # 归一化
        
        weight_change_lateral_inhibitory = self.alpha_I * torch.ger(postsynaptic_activity, postsynaptic_activity)
        self.lateral_weights_inhibitory.data += weight_change_lateral_inhibitory
        self.lateral_weights_inhibitory.data /= self.lateral_weights_inhibitory.data.sum(0, keepdim=True)  # 归一化



if __name__ == "__main__":
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--type', type=str, default='video', help="Input images source. 'video' for video, 'folder' for images in the 'images' subfolder.")
    parser.add_argument('--video_name', type=str, default='video.mp4', help='video name')

    args = parser.parse_args()

    if args.type == 'video':
        images = frogtools.get_images_from_video(args.video_name, IMAGE_WIDTH, IMAGE_HEIGHT, output_format='torch')
    elif args.type == 'folder':
        images = frogtools.get_images_from_folder('images', IMAGE_WIDTH, IMAGE_HEIGHT, output_format='torch')
    
    # 初始化前向连接和侧向连接的权重
    
    
    # 遍历所有图片
    for _ in range(len(images)):
        current_image = images[_]

        