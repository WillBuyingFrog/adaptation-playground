import os
import cv2
import numpy as np
import torch
from PIL import Image



def get_images_from_folder(folder_path, image_width, image_height, output_format='torch'):

    # 检查输入参数正确性
    assert output_format in ['torch', 'numpy'] 

    # 文件夹路径
    # 路径是这里固定的，具体是用os.join得出当前文件夹路径下的images文件夹
    folder_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), folder_path)

    # 获取文件夹中的所有文件
    file_list = os.listdir(folder_path)


    # 遍历文件夹中的每个文件
    for file_name in file_list:

        # 构建图像文件的完整路径
        file_path = os.path.join(folder_path, file_name)
        
        # 使用PIL库打开图像文件
        image = Image.open(file_path)

        # 强制转换图像大小
        image = image.resize((image_width, image_height))
        
        # 将图像转换为numpy数组并添加到列表中
        image = np.array(image)
        images.append(image)

    if output_format == 'numpy':
        ret_images = np.array(images)
    elif output_format == 'torch':
        ret_images = torch.tensor(images).permute(0, 3, 1, 2).float()
    
    return ret_images


def get_images_from_video(video_name, image_width, image_height, output_format='torch'):
    
    # 检查输入参数正确性
    assert output_format in ['torch', 'numpy'] 

    # 视频文件路径
    # 路径是这里固定的，具体是用os.join得出当前文件夹路径下的video文件夹下的video.mp4文件
    video_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'video', video_name)

    # 使用OpenCV库打开视频文件
    video = cv2.VideoCapture(video_path)

    # 返回的图像列表
    ret_images = []

    # 读取视频文件的每一帧
    while True:
        # 读取下一帧
        ret, frame = video.read()

        # 如果没有下一帧，则退出循环
        if not ret:
            break
        
        # 强制转换图像大小
        frame = cv2.resize(frame, (image_width, image_height))

        # 将图像转换为numpy数组并添加到列表中
        ret_images.append(frame)

    if output_format == 'numpy':
        # 将图像列表转换为numpy数组并返回
        ret_images = np.array(ret_images)
    elif output_format == 'torch':
        # 将图像列表转换为torch数组并返回
        ret_images = np.array(ret_images)
        ret_images = torch.tensor(ret_images).permute(0, 3, 1, 2).float()

    return ret_images