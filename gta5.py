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


class Gta5(Dataset):
    def __init__(self,mode,setaug):
        super(Gta5, self).__init__()
        assert mode in ('train', 'val','train_fda','val_fda')
        self.mode = mode
        self.setaug = setaug
        self.ignore_lb = 255


        ## parse img directory
        if mode=='train_fda' or mode == 'val_fda':            
            self.imgs = {}
            imgnames = []
            impth = osp.join(rootpth, 'GTA5/images', mode)
            folders = os.listdir(impth)
            for fd in folders:
                fdpth = osp.join(impth, fd)
                im_names = os.listdir(fdpth)
                names = [el.replace('.png', '') for el in im_names]
                names = [el.replace('da_fda_', '') for el in names]
                impths = [osp.join(fdpth, el) for el in im_names]
                imgnames.extend(names)
                self.imgs.update(dict(zip(names, impths)))
        else:
            self.imgs = {}
            imgnames = []
            impth = osp.join(rootpth, 'GTA5/images', mode)
            folders = os.listdir(impth)
            for fd in folders:
                fdpth = osp.join(impth, fd)
                im_names = os.listdir(fdpth)
                names = [el.replace('.png', '') for el in im_names]
                impths = [osp.join(fdpth, el) for el in im_names]
                imgnames.extend(names)
                self.imgs.update(dict(zip(names, impths)))

        ## parse gt directory
        if mode == 'train_fda' or mode == 'val_fda':
            self.labels = {}
            gtnames = []
            gtpth = osp.join(rootpth, 'GTA5/labels__', mode)
            folders = os.listdir(gtpth)
            for fd in folders:
                fdpth = osp.join(gtpth, fd)
                lbnames = os.listdir(fdpth)
                lbnames = [el for el in lbnames if 'da_fda_' in el]
                names = [el.replace('.png', '') for el in lbnames]
                names = [el.replace('da_fda_', '') for el in names]
                lbpths = [osp.join(fdpth, el) for el in lbnames]
                gtnames.extend(names)
                self.labels.update(dict(zip(names, lbpths)))
        else:
            self.labels = {}
            gtnames = []
            gtpth = osp.join(rootpth, 'GTA5/labels__', mode)
            folders = os.listdir(gtpth)
            for fd in folders:
                fdpth = osp.join(gtpth, fd)
                lbnames = os.listdir(fdpth)
                lbnames = [el for el in lbnames if 'labelTrainIds' in el]
                names = [el.replace('_labelTrainIds.png', '') for el in lbnames]
                lbpths = [osp.join(fdpth, el) for el in lbnames]
                gtnames.extend(names)
                self.labels.update(dict(zip(names, lbpths)))

        self.imnames = imgnames
        self.len = len(self.imnames)
        print(f'{self.mode}:{ self.len}')        
        assert set(imgnames) == set(gtnames)
        assert set(self.imnames) == set(self.imgs.keys())
        assert set(self.imnames) == set(self.labels.keys())        


        ## pre-processing
        self.to_tensor = transforms.Compose([
            transforms.Resize([512,1024]),
            transforms.ToTensor(),
            transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
            ])  
        
                         
        self.trans_train = transforms.Compose([
        transforms.RandomApply([transforms.ColorJitter( brightness = 0.5,  contrast = 0.5,saturation = 0.5)], p = 0.5 ),  
        ])
            
    
    def __getitem__(self, idx):
        fn  = self.imnames[idx]
        impth = self.imgs[fn]
        lbpth = self.labels[fn]
        img = Image.open(impth).convert('RGB')
        label = Image.open(lbpth)
        label = label.resize([1024,512], Image.NEAREST)
        if self.setaug == True and (self.mode == 'train' or self.mode == 'train_fda') : 
            img = self.trans_train(img)
        img = self.to_tensor(img)
        label = np.array(label).astype(np.int64)[np.newaxis, :]
        return img, label    
    

    def __len__(self):
        return self.len

# if __name__ == '__main__':
#     dataloader = DataLoader(Gta5('val_fda',1),
#                       batch_size=2,
#                       shuffle=False,
#                       num_workers=1,
#                       pin_memory=False,
#                       drop_last=True)
    
#     for images, labels in dataloader:
#         for i in range(len(images)):
#             image = transforms.ToPILImage()(images[i])
#         plt.subplot(1,2,1)
#         plt.imshow(image)
#         plt.subplot(1,2,2)           
#         plt.imshow(labels[i].squeeze(),'gray')      
#         break