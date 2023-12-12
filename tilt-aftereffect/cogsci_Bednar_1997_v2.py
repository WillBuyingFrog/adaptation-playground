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
# 模型参数
# ---------

# 定义皮层模型的神经元个数，神经元为2维排布
# 原始论文的神经元个数为192*192
CORTIAL_WIDTH_COUNT = 192
CORTIAL_HEIGHT_COUNT = 192

# 每个神经元在图像上的感受野宽高
# 原始论文的感受野大小为24*24
RECEPTIVE_FIELD_WIDTH = 24
RECEPTIVE_FIELD_HEIGHT = 24


