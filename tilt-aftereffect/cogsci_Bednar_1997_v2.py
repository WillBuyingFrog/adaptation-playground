import os
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt

import frogtools

# --------
# 调试参数
# --------
DEBUG_SWITCH = True



# --------
# 图像参数
# --------

# 定义输入图像的宽高
# 原始论文输入图像宽高为36，并且为灰度图片
IMAGE_HEIGHT = 36
IMAGE_WIDTH = 36
IMAGE_CHANNELS = 1

# 输入的图像
images = []

# ---------
# 神经元位置参数
# ---------

# 定义皮层模型的神经元个数，神经元为2维排布
# 原始论文的神经元个数为192*192
CORTEX_X_COUNT = 192
CORTEX_Y_COUNT = 192

# 每个神经元在图像上的感受野宽高
# 原始论文的感受野大小为24*24
RECEPTIVE_FIELD_WIDTH = 24
RECEPTIVE_FIELD_HEIGHT = 24

# 计算得出最左上角、最右下角的神经元的感受野起始坐标
CORTEX_RF_MINX = 0
CORTEX_RF_MINY = 0
CORTEX_RF_MAXX = IMAGE_WIDTH - RECEPTIVE_FIELD_WIDTH
CORTEX_RF_MAXY = IMAGE_HEIGHT - RECEPTIVE_FIELD_HEIGHT

# 计算得出每个神经元的感受野起始坐标
cortex_rf_start = torch.zeros((CORTEX_X_COUNT, CORTEX_Y_COUNT, 2))
_delta_x = (CORTEX_RF_MAXX - CORTEX_RF_MINX) / (CORTEX_X_COUNT - 1)
_delta_y = (CORTEX_RF_MAXY - CORTEX_RF_MINY) / (CORTEX_Y_COUNT - 1)
for i in range(CORTEX_X_COUNT):
    for j in range(CORTEX_Y_COUNT):
        _rf_start_x = CORTEX_RF_MINX + i * _delta_x
        _rf_start_y = CORTEX_RF_MINY + j * _delta_y
        cortex_rf_start[i, j, 0] = round(_rf_start_x)
        cortex_rf_start[i, j, 1] = round(_rf_start_y)


# --------
# 各类神经元连接的mask
# --------

# 前向连接mask
affterent_mask = torch.zeros((RECEPTIVE_FIELD_WIDTH, RECEPTIVE_FIELD_HEIGHT))
_grid_x, _grid_y = torch.meshgrid(torch.arange(RECEPTIVE_FIELD_WIDTH), torch.arange(RECEPTIVE_FIELD_HEIGHT))
_dist = torch.sqrt((_grid_x - RECEPTIVE_FIELD_WIDTH / 2) ** 2 + (_grid_y - RECEPTIVE_FIELD_HEIGHT / 2) ** 2)  # 先不考虑奇数情况
affterent_mask[_dist <= 6.0] = 1

# 侧向连接mask
# 按照论文原文，侧向连接mask会在训练过程中调整，所以先写一个函数
def get_lateral_mask(required_dist, lateral_type='excitatory'):

    assert lateral_type in ['excitatory', 'inhibitory']
    
    if lateral_type == 'excitatory':
        lateral_mask = torch.zeros((39, 39))
        grid_x, grid_y = torch.meshgrid(torch.arange(39), torch.arange(39))
        dist = torch.sqrt((grid_x - 19) ** 2 + (grid_y - 19) ** 2)
        lateral_mask[dist <= required_dist] = 1
    else:
        lateral_mask = torch.zeros((95, 95))
        grid_x, grid_y = torch.meshgrid(torch.arange(95), torch.arange(95))
        dist = torch.sqrt((grid_x - 47) ** 2 + (grid_y - 47) ** 2)
        lateral_mask[dist <= required_dist] = 1
    return lateral_mask

# 初始的侧向连接mask
excitatory_lateral_mask = get_lateral_mask(19, lateral_type='excitatory')
inhibitory_lateral_mask = get_lateral_mask(47, lateral_type='inhibitory')


# 此外对每个神经元而言，在计算侧向连接时，会存在上述mask合法但位置本身不合法的情况，
# 比如(0,0)位置的神经元的侧向连接mask就只有右下四分之一的部分是有效的，因为剩下来的mask都对应了负坐标
# 所以需要一个mask来标记这些位置。节省内存，这里用short类型
cortex_lateral_mask = torch.zeros((CORTEX_X_COUNT, CORTEX_Y_COUNT, CORTEX_X_COUNT, CORTEX_Y_COUNT)).short()
for i in range(CORTEX_X_COUNT):
    for j in range(CORTEX_Y_COUNT):
        _mask = torch.zeros((CORTEX_X_COUNT, CORTEX_Y_COUNT)).short()
        _grid_x, _grid_y = torch.meshgrid(torch.arange(i - 19, i + 20), torch.arange(j - 19, j + 20))
        _mask[(_grid_x >= 0) & (_grid_x < CORTEX_X_COUNT) & (_grid_y >= 0) & (_grid_y < CORTEX_Y_COUNT)] = 1
        cortex_lateral_mask[i, j] = _mask

# --------
# 模型实现
# --------

class CortexModel(nn.Module):
    def __init__(self, cortex_x_count, cortex_y_count, image_channels,
                 afferent_weights, ex_lateral_weight_init, in_lateral_weight_init,
                 afferent_mask, ex_lateral_mask, in_lateral_mask, cortex_lateral_mask,
                 cortex_rf_start, rf_width, rf_height,
                 ex_lateral_radius, in_lateral_radius,
                 gamma_e, gamma_i, alpha_A, alpha_E, alpha_I):
        super(CortexModel, self).__init__()

        self.cortex_x_count = cortex_x_count
        self.cortex_y_count = cortex_y_count
        self.image_channels = image_channels
        
        self.afferent_weights = nn.Parameter(afferent_weights)
        self.ex_lateral_weight = nn.Parameter(ex_lateral_weight_init)
        self.in_lateral_weight = nn.Parameter(in_lateral_weight_init)

        self.afferent_mask = afferent_mask
        self.excitatory_lateral_mask = ex_lateral_mask
        self.inhibitory_lateral_mask = in_lateral_mask
        # self.cortex_lateral_mask = cortex_lateral_mask

        self.ex_lateral_radius = ex_lateral_radius
        self.in_lateral_radius = in_lateral_radius

        self.cortex_rf_start = cortex_rf_start
        self.rf_width = rf_width
        self.rf_height = rf_height

        self.gamma_e = gamma_e
        self.gamma_i = gamma_i
        self.alpha_A = alpha_A
        self.alpha_E = alpha_E
        self.alpha_I = alpha_I

        self.activation = torch.zeros((self.image_channels, cortex_x_count, cortex_y_count)).float()

    
    def forward(self, input):
        
        new_activation = torch.zeros((self.image_channels, 47+self.cortex_x_count+47,
                                     47+self.cortex_y_count+47)).float()
        
        old_activation_padding = torch.zeros_like(new_activation)
        old_activation_padding[:,47:47+self.cortex_x_count, 47:47+self.cortex_y_count] = self.activation.copy()
        
        # 计算前向输出
        for i in range(self.cortex_x_count):
            for j in range(self.cortex_y_count):
                receptive_field = input[:, self.cortex_rf_start[i, j, 0]:self.cortex_rf_start[i, j, 0] + RECEPTIVE_FIELD_WIDTH,
                                        self.cortex_rf_start[i, j, 1]:self.cortex_rf_start[i, j, 1] + RECEPTIVE_FIELD_HEIGHT]
                new_activation[:,i+47,j+47] = torch.sum(receptive_field * self.afferent_weights[:,i,j] * self.afferent_mask)
        

        # 计算兴奋侧向连接输出
        excitatory_lateral_activation = torch.zeros((self.image_channels, self.cortex_x_count, self.cortex_y_count)).float()
        for i in range(self.cortex_x_count):
            for j in range(self.cortex_y_count):
                excitatory_lateral_activation[:,i,j] = old_activation_padding[:,i-19:i+19,j-19:j+19] * self.ex_lateral_weight * self.excitatory_lateral_mask
        
        inhibitory_lateral_activation = torch.zeros((self.image_channels, self.cortex_x_count, self.cortex_y_count)).float()
        for i in range(self.cortex_x_count):
            for j in range(self.cortex_y_count):
                inhibitory_lateral_activation[:,i,j] = old_activation_padding[:,i-47:i+47,j-47:j+47] * self.in_lateral_weight * self.inhibitory_lateral_mask

        
        new_activation[:, 47:47+self.cortex_x_count, 47:47+self.cortex_y_count] = new_activation[:, 47:47+self.cortex_x_count, 47:47+self.cortex_y_count]
        + self.gamma_e * excitatory_lateral_activation - self.gamma_i * inhibitory_lateral_activation

        self.activation = new_activation[:, 47:47+self.cortex_x_count, 47:47+self.cortex_y_count]

        return self.activation
    

    def update_weigths():
        pass