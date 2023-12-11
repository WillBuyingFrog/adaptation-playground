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
# 原始论文输入图像宽高为192，并且为灰度图片
IMAGE_HEIGHT = 192
IMAGE_WIDTH = 192
IMAGE_CHANNELS = 1

# 输入的图像
images = []

# ---------
# 模型参数
# ---------

# 定义皮层模型的神经元个数，神经元为2维排布
# 原始论文的神经元个数为36*36
CORTIAL_WIDTH_COUNT = 36
CORTIAL_HEIGHT_COUNT = 36

# 每个神经元在图像上的感受野宽高
# 原始论文的感受野大小为24*24
RECEPTIVE_FIELD_WIDTH = 24
RECEPTIVE_FIELD_HEIGHT = 24

# 计算得出宽度方向上，每个神经元在输入图片上，感受野的起始宽度坐标
# 图片的最小坐标为0，最大坐标为IMAGE_WIDTH-1
# CORTIAL_WIDTH个神经元的起始坐标需要平摊在宽度坐标上，第一个神经元的起始坐标为0，第二个神经元的起始坐标为k，
# 第三个神经元的起始坐标为2k，以此类推，直到最后一个神经元的感受野起始坐标为(RECEPTIVE_FIELD_WIDTH-1)*k
# 计算出不考虑取整的k
k1 = (IMAGE_WIDTH - RECEPTIVE_FIELD_WIDTH) / (CORTIAL_WIDTH_COUNT - 1)
# 保存横向上每个神经元的感受野起始坐标
receptive_field_start_width = []
# 计算出不考虑取整情况下的起始坐标
for i in range(CORTIAL_WIDTH_COUNT):
    temp_start_width = i * k1
    # 四舍五入取整
    temp_start_width = round(temp_start_width)
    # 检查是否越界，如果越界则取最大值
    temp_start_width = min(temp_start_width, IMAGE_WIDTH - RECEPTIVE_FIELD_WIDTH)
    receptive_field_start_width.append(temp_start_width)


# 对高度方向上的感受野起始坐标也做同样计算
k2 = (IMAGE_HEIGHT - RECEPTIVE_FIELD_HEIGHT) / (CORTIAL_HEIGHT_COUNT - 1)
receptive_field_start_height = []
for i in range(CORTIAL_HEIGHT_COUNT):
    temp_start_height = i * k2
    temp_start_height = round(temp_start_height)
    temp_start_height = min(temp_start_height, IMAGE_HEIGHT - RECEPTIVE_FIELD_HEIGHT)
    receptive_field_start_height.append(temp_start_height)

# 将保存下来的坐标都转换成pytorch里的向量
receptive_field_x = torch.Tensor(receptive_field_start_width)
receptive_field_y = torch.Tensor(receptive_field_start_height)

# --------
# 模型实现
# --------


class CortexNetwork(nn.Module):
    def __init__(self, cortial_x_count, cortial_y_count, afferent_weights,
                 ex_lateral_weight_init, in_lateral_weight_init, gamma_e, gamma_i, alpha_A, alpha_E, alpha_I):
        super(CortexNetwork, self).__init__()
        self.cortial_x_count = cortial_x_count
        self.cortial_y_count = cortial_y_count
        
        # 前向连接权重的形状是(1, 宽方向神经元数量, 高方向神经元数量, 感受野宽, 感受野高)
        self.afferent_weights = nn.Parameter(afferent_weights)


        self.excitatory_latertal_weights = nn.Parameter(ex_lateral_weight_init)  # 初始化兴奋性侧向权重
        self.inhibitory_lateral_weights = nn.Parameter(in_lateral_weight_init)  # 初始化抑制性侧向权重
        self.gamma_e = gamma_e  # 兴奋性侧向连接的缩放因子
        self.gamma_i = gamma_i  # 抑制性侧向连接的缩放因子
        
        self.alpha_A = alpha_A  # 传入连接的学习率
        self.alpha_E = alpha_E  # 兴奋性连接的学习率
        self.alpha_I = alpha_I  # 抑制性连接的学习率

    
    def forward(self, input, prev_activity):
        # 计算传入激活
        
        # 传入激活的大小为 通道数 * 神经元宽方向个数 * 神经元高方向个数
        # 本论文涉及的模型采用灰度图像为输入，因此通道数为1
        afferent_activation = torch.zeros(IMAGE_CHANNELS, self.cortial_x_count, self.cortial_y_count)


        for i in range(CORTIAL_WIDTH_COUNT):
            for j in range(CORTIAL_HEIGHT_COUNT):
                receptive_field = input[:, receptive_field_x[i]:receptive_field_x[i]+RECEPTIVE_FIELD_WIDTH,
                                         receptive_field_y[j]:receptive_field_y[j]+RECEPTIVE_FIELD_HEIGHT]
                # 在这里，每个神经元的"加权平均池化"参数是独立的，因此需要调用afferent_weights[:,i,j]来获取每个神经元在其感受野中的加权平均池化参数
                # 然后再计算出afferent_activation[i][j]的值
                afferent_activation[:,i,j] = torch.sum(receptive_field * self.afferent_weights[:,i,j])

            

        # 计算侧向激活
        # 兴奋侧向激活权重的形状为 (神经元宽方向个数 * 神经元高方向个数 * 神经元宽方向个数 * 神经元高方向个数)
        # 兴奋侧向激活权重表示为E_ch_i_j_m_n，代表第ch个通道下，对(i,j)位置的神经元来说，(m,n)与他的兴奋侧向连接的权重
        # 兴奋侧向激活结果的形状为（神经元宽方向个数 * 神经元高方向个数）
        # 兴奋侧向激活结果表示为 EI_ch_m_n，代表第ch个通道下，(m,n)位置的神经元的兴奋侧向连接的结果，这个结果后面会加在神经元前向输出上，作为总激活的一部分
        excitatory_lateral_activation = torch.zeros(IMAGE_CHANNELS, self.cortial_x_count, self.cortial_y_count)
        for i in range(CORTIAL_WIDTH_COUNT):
            for j in range(CORTIAL_HEIGHT_COUNT):
                excitatory_lateral_activation[:,i,j] = torch.sum(prev_activity[:,i,j] * self.excitatory_latertal_weights[:,i,j])

        # 抑制侧向激活权重的计算方法同理
        inhibitory_lateral_activation = torch.zeros(IMAGE_CHANNELS, self.cortial_x_count, self.cortial_y_count)
        for i in range(CORTIAL_WIDTH_COUNT):
            for j in range(CORTIAL_HEIGHT_COUNT):
                inhibitory_lateral_activation[i,j] = torch.sum(prev_activity[:,i,j] * self.inhibitory_lateral_weights[:,i,j])

        
        # 计算总激活
        total_activation = afferent_activation + self.gamma_e * excitatory_lateral_activation + self.gamma_i * inhibitory_lateral_activation
        activity = self.sigmoid_approximation(total_activation)
        
        return activity
    
    def sigmoid_approximation(self, x):
        # 实现逐段线性Sigmoid近似函数
        return torch.clamp(x, min=0)  # 这里只是一个简单的ReLU作为示例
    
    # TODO 这个函数写得不对，需要重写
    def update_weights(self, presynaptic_activity, postsynaptic_activity):
        # Hebbian权重更新规则
        weight_change_afferent = self.alpha_A * torch.ger(presynaptic_activity, postsynaptic_activity)
        self.afferent_weights.data += weight_change_afferent
        self.afferent_weights.data /= self.afferent_weights.data.sum(0, keepdim=True)  # 归一化
        
        



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

        