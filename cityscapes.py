#!/usr/bin/python
# -*- encoding: utf-8 -*-
from torch.utils.data import Dataset,DataLoader
import torch
import torchvision.transforms as transforms
import os.path as osp
import os
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt


rootpth = './data/'


class CityScapes(Dataset):
    def __init__(self, mode):
        super(CityScapes, self).__init__()
        assert mode in ('train', 'val')
        self.mode = mode
        self.ignore_lb = 255
       
        ## parse img directory
        self.imgs = {}
        imgnames = []
        impth = osp.join(rootpth, 'Cityscapes/images', mode)
        folders = os.listdir(impth)
        for fd in folders:
            fdpth = osp.join(impth, fd)
            im_names = os.listdir(fdpth)
            names = [el.replace('_leftImg8bit.png', '') for el in im_names]
            impths = [osp.join(fdpth, el) for el in im_names]
            imgnames.extend(names)
            self.imgs.update(dict(zip(names, impths)))

        ## parse gt directory
        self.labels = {}
        gtnames = []
        gtpth = osp.join(rootpth, 'Cityscapes/gtFine', mode)
        folders = os.listdir(gtpth)
        for fd in folders:
            fdpth = osp.join(gtpth, fd)
            lbnames = os.listdir(fdpth)
            lbnames = [el for el in lbnames if 'labelTrainIds' in el]
            names = [el.replace('_gtFine_labelTrainIds.png', '') for el in lbnames]
            lbpths = [osp.join(fdpth, el) for el in lbnames]
            gtnames.extend(names)
            self.labels.update(dict(zip(names, lbpths)))

        self.imnames = imgnames
        self.len = len(self.imnames)
        print(f'{self.mode}:{self.len}')        
        assert set(imgnames) == set(gtnames)
        assert set(self.imnames) == set(self.imgs.keys())
        assert set(self.imnames) == set(self.labels.keys())

        ## pre-processing
        self.to_tensor = transforms.Compose([
            transforms.Resize([512,1024]),
            transforms.ToTensor(),
            transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
            ])

        
    def convert_labels(self, label):
        for k, v in self.lb_map.items():
            label[label == k] = v
        return label
    
    
    def __getitem__(self, idx):
        fn  = self.imnames[idx]
        impth = self.imgs[fn]
        lbpth = self.labels[fn]
        img = Image.open(impth).convert('RGB')
        label = Image.open(lbpth)
        label = label.resize([1024,512], Image.NEAREST)
        img = self.to_tensor(img)
        label = np.array(label).astype(np.int64)[np.newaxis, :]
        
        return img, label

    def __len__(self):
        return self.len
    
    
# if __name__ == '__main__':
#     dataloader = DataLoader(CityScapes('val'),
#                       batch_size=1,
#                       shuffle=False,
#                       num_workers=1,
#                       pin_memory=False,
#                       drop_last=True)
 
#     for images, labels in dataloader:
#         for i in range(len(images)):
#             image = transforms.ToPILImage()(images[i])
#             plt.subplot(1,2,1)
#             plt.imshow(image)
#             plt.subplot(1,2,2)           
#             plt.imshow(labels[i].squeeze(),'gray')      
#         break
