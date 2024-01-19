# -*- coding: utf-8 -*-
"""
Created on Thu Jan 11 11:54:16 2024

@author: liuzh
"""

import os
import shutil
import random

# 设置随机种子以确保每次运行时分割结果一致
random.seed(42)

# 定义原始数据集路径和目标路径
# original_data_path = './data/GTA5/images'
# train_data_path = './data\GTA5/images/train'
# val_data_path = './data/GTA5/images/val'

original_data_path = './GTA5/labels__'
train_data_path = './GTA5/labels__/train'
val_data_path = './GTA5/labels__/val'

# 确保目标文件夹存在，如果不存在则创建
os.makedirs(train_data_path, exist_ok=True)
os.makedirs(val_data_path, exist_ok=True)

# 获取所有图像文件的列表
image_files = [f for f in os.listdir(original_data_path) if f.endswith('.jpg') or f.endswith('.png')]

# 随机打乱图像列表
random.shuffle(image_files)

# 定义训练集和验证集的比例
train_ratio = 0.7  # 70% train，30% val

# 计算分割点
split_point = int(len(image_files) * train_ratio)

# 分割图像列表
train_images = image_files[:split_point]
val_images = image_files[split_point:]

# 将图像复制到训练集文件夹
for image in train_images:
    src_path = os.path.join(original_data_path, image)
    dst_path = os.path.join(train_data_path, image)
    shutil.copy(src_path, dst_path)

# 将图像复制到验证集文件夹
for image in val_images:
    src_path = os.path.join(original_data_path, image)
    dst_path = os.path.join(val_data_path, image)
    shutil.copy(src_path, dst_path)