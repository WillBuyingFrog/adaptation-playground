import numpy as np
import os
import cv2
import argparse
from PIL import Image


# 调试开关
DEBUG_SWITCH = True

# 定义图像宽高
IMAGE_WIDTH = 416
IMAGE_HEIGHT = 416

# 定义初始阈值
INITIAL_THRESHOLD = 140.0

# 用于存储图像的列表
images = []

# 定义单个比较区域大小
h, w = 100, 100

# 定义区域
regions = []
# 每个区域的起始左上角坐标之间需要分别隔开h和w
for i in range(0, IMAGE_HEIGHT - h, 25):
    for j in range(0, IMAGE_WIDTH - w, 25):
        regions.append((j, i, w, h))

# 用于存储图像每个区域变化阈值的numpy array，初始值为INITIAL_THRESHOLD
image_threshold = np.full((len(regions), ), INITIAL_THRESHOLD, dtype=np.float32)


def get_weighted_average_image(fr, to, save_figure=False):
    # 对过去至多10张照片计算加权平均
    # 从fr到to的照片，越新的照片权重越大
    image_weights = np.array([5, 3, 2.5, 2, 1.5, 1.25, 1, 0.75, 0.5, 0.25])
    
    # 需要考虑到照片数量不足10张的情况，需要根据实际传入的照片总数计算实际用到的权重的总和
    weight_sum = np.sum(image_weights[:to - fr])

    # 使用权重对图像进行加权平均
    # print(image_weights[:to - fr].shape)
    weighted_average_image = np.average(images[fr:to], axis=0, weights=image_weights[:to - fr] / weight_sum)

    if save_figure:
        # 保存这张平均值图片到results子文件夹中
        cv2.imwrite(f"results/average_{fr}_{to}.jpg", weighted_average_image)
    return weighted_average_image



def get_average_image(fr, to, save_figure=False):

    # 获取image_list[fr:to]的平均值
    avg_image =  np.average(images[fr:to], axis=0)
    if save_figure:
        # 保存这张平均值图片到results子文件夹中
        cv2.imwrite(f"results/average_{fr}_{to}.jpg", avg_image)
    return avg_image


def get_images_from_folder():
    # 文件夹路径
    # 路径是这里固定的，具体是用os.join得出当前文件夹路径下的images文件夹
    folder_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'images')

    # 获取文件夹中的所有文件
    file_list = os.listdir(folder_path)



    # 遍历文件夹中的每个文件
    for file_name in file_list:

        # 构建图像文件的完整路径
        file_path = os.path.join(folder_path, file_name)
        
        # 使用PIL库打开图像文件
        image = Image.open(file_path)

        # 强制转换图像大小
        image = image.resize((IMAGE_WIDTH, IMAGE_HEIGHT))
        
        # 将图像转换为numpy数组并添加到列表中
        image = np.array(image)
        images.append(image)

    # 将图像列表转换为numpy数组并返回
    images = np.array(images)


def get_images_from_video(video_name):

    global images

    # 视频文件路径
    # 路径是这里固定的，具体是用os.join得出当前文件夹路径下的video文件夹下的video.mp4文件
    video_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'video', video_name)

    # 使用OpenCV库打开视频文件
    video = cv2.VideoCapture(video_path)

    # 读取视频文件的每一帧
    while True:
        # 读取下一帧
        ret, frame = video.read()

        # 如果没有下一帧，则退出循环
        if not ret:
            break
        
        # 强制转换图像大小
        frame = cv2.resize(frame, (IMAGE_WIDTH, IMAGE_HEIGHT))

        # 将图像转换为numpy数组并添加到列表中
        images.append(frame)

    images = np.array(images)

if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='Simpliest sample of visual adaptation')

    parser.add_argument('--mode', type=str, default='folder', help='Whether to read images from folder or a video')
    parser.add_argument('--video-name', type=str, default='video.mp4', help='The name of the video file')

    args = parser.parse_args()

    if args.mode == 'folder':
        get_images_from_folder()
    elif args.mode == 'video':
        get_images_from_video(args.video_name)


    # 获取图像的数量
    image_count = images.shape[0]

    for i in range(1, image_count):

        # 计算至多过去十张图片的平均值
        current_average = get_weighted_average_image(max(i-10, 0), i, i % 10 == 0)

        diffs = np.zeros((len(regions), ), dtype=np.float32)

        temp_im = images[i].copy()
        counter = 0
        for _ in range(len(regions)):
            xx, yy, ww, hh = regions[_]
            diffs[_] = np.sum(np.abs(images[i, yy:yy+hh, xx:xx+ww] - current_average[yy:yy+hh, xx:xx+ww])) / (ww * hh)
            # 每500个region输出一次
            # if _ % 500 == 0:
            #     print(f"Region #{_}, location ({xx}, {yy}), diff: {diffs[yy, xx]}")
            # 如果当前region的变化超过了阈值，那么就保存一张图片，图片里是经过缩放之后的图片，上面用红框框出对应变化超过阈值的区域
            if diffs[_] > image_threshold[_]:
                # 先画红框再保存图片
                temp_im = cv2.rectangle(temp_im, (xx, yy), (xx+ww, yy+hh), (0, 0, 255), 1)
                counter += 1
        if i % 10 == 0:
            print(f"Round {i} has {counter} regions beyond threshold")
            cv2.imwrite(f"results/round_{i}_sampled.jpg", temp_im)
            cv2.imwrite(f"results/round_{i}_initial.jpg", images[i])

        # 得出diffs中所有比阈值大的数值的索引
        indexes = np.where(diffs > image_threshold)

        # 更新阈值
        # 对所有diff超过阈值的区域，恢复这些区域的阈值为INITIAL_THRESHOLD
        image_threshold[indexes] = INITIAL_THRESHOLD
        # 对剩下没有超过阈值的区域，减小这些区域的阈值
        image_threshold[~indexes[0]] /= 1.2
