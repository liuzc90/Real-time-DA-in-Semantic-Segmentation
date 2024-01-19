# -*- coding: utf-8 -*-
"""
Created on Wed Dec 27 20:38:39 2023

@author: liuzh
"""

import cv2
import numpy as np
from PIL import Image
import os
import glob

def rgb_to_gray(rgb_image):
    color_mapping = {
        (128, 64, 128):0,
        (244, 35, 232):1,
        (70, 70, 70):2,
        (102, 102, 156):3,
        (190, 153, 153):4,
        (153, 153, 153):5,
        (250, 170, 30):6,
        (220, 220, 0):7,
        (107, 142, 35):8,
        (152, 251, 152):9,
        (70, 130, 180):10,
        (220, 20, 60):11,
        (255, 0, 0):12,
        (0, 0, 142):13,
        (0, 0, 70):14,
        (0, 60, 100):15,
        (0, 80, 100):16,
        (0, 0, 230):17,
        (119, 11, 32):18
    }

    # Create a color map for efficient lookup
    color_map = {color: value for color, value in color_mapping.items()}

    # Create an array of shape (height, width) filled with 255
    gray_array = np.full(rgb_image.shape[:2], 255, dtype=np.uint8)

    # Use broadcasting to compare the entire image with the color map
    for color, value in color_map.items():
        mask = np.all(rgb_image == np.array(color), axis=-1)
        gray_array[mask] = value

    # Create a Pillow image object
    gray_image = Image.fromarray(gray_array)

    return gray_image

def rgb_to_gray_and_save(input_folder, output_folder, label_suffix="_labelTrainIds"):
    # 获取输入文件夹中所有的png图像文件
    image_files = glob.glob(os.path.join(input_folder, '*.png'))

    # 确保输出文件夹存在
    os.makedirs(output_folder, exist_ok=True)

    for i, image_file in enumerate(image_files):
        # 使用Pillow打开RGB图像
        image_bgr = cv2.imread(image_file)
        image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)

        # 获取图像的文件名（不带路径）
        base_name = os.path.basename(image_file)

        # 将RGB图像转换为灰度图像
        gray_image = rgb_to_gray(image_rgb)

        print(f"Processing image {i + 1}/{len(image_files)}")

        # 生成保存的文件路径
        output_file = os.path.join(output_folder, f"{os.path.splitext(base_name)[0]}{label_suffix}.png")

        # 保存灰度图像
        cv2.imwrite(output_file, np.array(gray_image))

if __name__ == "__main__":
    # 输入文件夹和输出文件夹的路径
    input_folder_path = "./GTA5/labels"
    output_folder_path = "./GTA5/labels_new"

    # 转换RGB图像为灰度图像并保存
    rgb_to_gray_and_save(input_folder_path, output_folder_path)