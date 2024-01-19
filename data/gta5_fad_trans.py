# -*- coding: utf-8 -*-
"""
Created on Thu Jan 11 16:40:03 2024

@author: liuzh
"""

from torch.utils.data import Dataset,DataLoader
import torch
from torch.utils import data
import torchvision.transforms as transforms
import os.path as osp
import os
from PIL import Image
import numpy as np
# from transform import *
import json
from FDA import FDA_source_to_target_np, toimage
import skimage.color as color
import matplotlib.pyplot as plt
import random

root = './GTA5/'
transformation='FDA'
target_folder='./Cityscapes/images/train'
max_iters=None, 
crop_size=(1024, 512) 
mean=(104.00698793, 116.66876762, 122.67891434)
scale=True
ignore_label=255

# train set
# imgs_path = osp.join(root,"images/train/gta/" + "train.txt") 
# labels_path = osp.join(root,"labels__/train/gta/""labels.txt")
# val set
imgs_path = osp.join(root,"images/val/gta2/" + "images_val.txt") 
labels_path = osp.join(root,"labels__/val/gta2/"+"labels_val.txt")        

img_ids = [i_id.strip() for i_id in open(imgs_path)]
label_ids = [i_id.strip() for i_id in open(labels_path)]
   

files = []

id_to_trainid = {7: 0, 8: 1, 11: 2, 12: 3, 13: 4, 17: 5,
                      19: 6, 20: 7, 21: 8, 22: 9, 23: 10, 24: 11, 25: 12,
                      26: 13, 27: 14, 28: 15, 31: 16, 32: 17, 33: 18}

for ind in range(len(img_ids)):
        img_file = osp.join(root, "images/val/gta2/" +  img_ids[ind])
        label_file = osp.join(root, "labels__/val/gta2/" + label_ids[ind])
        files.append({
            "img": img_file,
            "label": label_file,
            "name": img_ids[ind]
        })
           
#GET TARGET IF TRANSFORMATION ON SOURCE IS APPLIED
img_ids_target = [i_id.strip() for i_id in open(osp.join(target_folder, "train.txt"))]
       

filestarget = []
for name in img_ids_target:
    image_path = osp.join (target_folder + '/' + name)
    names=name.split('/')[1].split('_')
    name = names[0]+'_'+names[1]+'_'+names[2]    
    filestarget.append({
        "image": image_path,
        "name": name
    })
  
file_len =  len(files)          


for index in range(file_len):
    print(f'{index}/{file_len}')
  
    datafiles = files[index]

    image = Image.open(datafiles["img"]).convert('RGB')
    label = Image.open(datafiles["label"])
    name = datafiles["name"]

    # resize
    image = image.resize(crop_size, Image.BICUBIC)
    label = label.resize(crop_size, Image.NEAREST)
    fda_img_name = "da_fda_" + name
    label_out_folder = './GTA5/labels__/val_fda/gta'   
    fda_label_path = os.path.join(label_out_folder, fda_img_name)  
    label.save(fda_label_path)




    image = np.asarray(image, np.float32)
    label = np.asarray(label, np.float32)

    # re-assign labels to match the format of Cityscapes
    label_copy = 255 * np.ones(label.shape, dtype=np.float32)
    for k, v in id_to_trainid.items():
        label_copy[label == k] = v

    #TRANSFORMATION
    random_number = random.randint(0, len(filestarget)-1)
    targetfiles=filestarget[random_number]
    targetimage=Image.open(targetfiles["image"]).convert('RGB')
    targetimage=targetimage.resize(crop_size,Image.BICUBIC)

    targetimage=np.asarray(targetimage, np.float32)
    image=image.transpose((2,0,1))
    targetimage=targetimage.transpose((2,0,1))
    src_in_trg = FDA_source_to_target_np(image, targetimage, L=0.001)
    image=src_in_trg.transpose((1,2,0))
    image=toimage(image, cmin=0.0, cmax=255.0)
    
    #save the FDA train image
    fda_img_name = "da_fda_" + name
    train_output_folder = './GTA5/images/val_fda/gta'
    fda_img_path = os.path.join(train_output_folder, fda_img_name)
    image.save(fda_img_path)

    
    

#%%
# import matplotlib.pyplot as plt
# plt.imshow(image)