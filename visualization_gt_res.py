# -*- coding: utf-8 -*-
"""
Created on Wed Jan 10 21:46:02 2024

@author: liuzh
"""

import numpy as np
import matplotlib.pyplot as plt
from model.model_stages import BiSeNet
import torch
from torch.utils.data import DataLoader
import logging
import argparse
import numpy as np
from utils import reverse_one_hot, compute_global_accuracy, fast_hist, per_class_iu
import os
from PIL import Image
import os.path as osp
import random
import torchvision.transforms as transforms
from torchvision import datasets
import os.path as osp
from torch.utils.data import Dataset


logger = logging.getLogger()


def gray_to_rgb(gray_img):

    # define clolor map
    color_mapping = {
        0:(128, 64, 128),
        1:(244, 35, 232),
        2:(70, 70, 70),
        3:(102, 102, 156),
        4:(190, 153, 153),
        5:(153, 153, 153),
        6:(250, 170, 30),
        7:(220, 220, 0),
        8:(107, 142, 35),
        9:(152, 251, 152),
        10:(70, 130, 180),
        11:(220, 20, 60),
        12:(255, 0, 0),
        13:(0, 0, 142),
        14:(0, 0, 70),
        15:(0, 60, 100),
        16:(0, 80, 100),
        17:(0, 0, 230),
        18:(119, 11, 32)
    }

    # create RGB image
    height, width = gray_img.shape
    rgb_image = np.zeros((height, width, 3), dtype=np.uint8)
    
    # mapping RGB image
    for i in range(height):
        for j in range(width):
            gray_value = gray_img[i, j]
            rgb_value = color_mapping.get(gray_value, (255, 255, 255))
            rgb_image[i, j, :] = np.array(rgb_value, dtype=np.uint8)
            
    return rgb_image
    

def val(args, model, dataloader):
    print('start test!')
    with torch.no_grad():
        model.eval()
        precision_record = []              
        hist = np.zeros((args.num_classes, args.num_classes))
        for i,( data, label) in enumerate(dataloader):
            print('Testing: {}/{}'.format(i+1, len(dataloader)))           
            
            label = label.type(torch.LongTensor)
            data = data.cuda()
            label = label.long().cuda()

            # get RGB predict image
            predict, _, _ = model(data)
            predict = predict.squeeze(0)
            predict = reverse_one_hot(predict)
            predict = np.array(predict.cpu())

            # get RGB label image
            label = label.squeeze()
            label = np.array(label.cpu())

            # compute per pixel accuracy
            precision = compute_global_accuracy(predict, label)
            hist += fast_hist(label.flatten(), predict.flatten(), args.num_classes)

            # there is no need to transform the one-hot array to visual RGB array
            # predict = colour_code_segmentation(np.array(predict), label_info)
            # label = colour_code_segmentation(np.array(label), label_info)
            precision_record.append(precision)

        precision = np.mean(precision_record)
        miou_list = per_class_iu(hist)
        miou = np.mean(miou_list)
        print('precision per pixel for test: %.3f' % precision)
        print('mIoU for validation: %.3f' % miou)
        print(f'mIoU per class: {miou_list}')

        return precision, miou, predict, label

def str2bool(v):
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Unsupported value encountered.')

def parse_args():
    parse = argparse.ArgumentParser()
    parse.add_argument('--mode', dest='mode', type=str, default='train' )
    parse.add_argument('--backbone', dest='backbone', type=str, default='CatmodelSmall') 
    parse.add_argument('--pretrain_path', dest='pretrain_path', type=str,default='./pretrain/STDCNet813M_73.91.tar')
    parse.add_argument('--use_conv_last',dest='use_conv_last',type=str2bool,default=False)                                                                        
    parse.add_argument('--num_epochs',type=int, default=50,help='Number of epochs to train for')                                              
    parse.add_argument('--epoch_start_i', type=int,default=0, help='Start counting epochs from this number')                                                                   
    parse.add_argument('--checkpoint_step',type=int, default=1,help='How often to save checkpoints (epochs)')                                                                    
    parse.add_argument('--validation_step',type=int,default=2,help='How often to perform validation (epochs)')                                                                     
    parse.add_argument('--crop_height',type=int,default=512,help='Height of cropped/resized input image to modelwork')                                                                     
    parse.add_argument('--crop_width', type=int, default=1024,help='Width of cropped/resized input image to modelwork')                                                                   
    parse.add_argument('--batch_size',type=int,default=2,help='Number of images in each batch')                                                                    
    parse.add_argument('--learning_rate',type=float,default=0.01,help='learning rate used for train')                                                                       
    parse.add_argument('--num_workers',type=int,default=2, help='num of workers')                                                                 
    parse.add_argument('--num_classes',type=int,default=19,help='num of object classes (with void)')                                                                   
    parse.add_argument('--cuda',type=str,default='1',help='GPU ids used for training')                                                                    
    parse.add_argument('--use_gpu',type=bool,default=True,help='whether to user gpu for training')                                                                     
    parse.add_argument('--save_model_path',type=str,default='./checkpoints/gta5/',help='path to save model')                                                                     
    parse.add_argument('--optimizer',type=str,default='adam', help='optimizer, support rmsprop, sgd, adam')                                                                    
    parse.add_argument('--loss', type=str,default='crossentropy',help='loss function')                                                                  
    return parse.parse_args()

def test(dataloader_test,weight_pth):
    args = parse_args()   
    ## dataset
    n_classes = args.num_classes
    mode = args.mode       
                
    ## model
    model = BiSeNet(backbone=args.backbone, n_classes=n_classes, pretrain_model=args.pretrain_path, use_conv_last=args.use_conv_last)
    
    # load best weights 
    checkpoint = torch.load(weight_pth)
    model.load_state_dict(checkpoint)
    if torch.cuda.is_available() and args.use_gpu:
        model = torch.nn.DataParallel(model).cuda()

    # final test
    return val(args, model, dataloader_test)
   
class testdata(Dataset):
    def __init__(self,test_set,random_index):
        super(testdata, self).__init__()
        
       
        if test_set == 'cityscapes':
            ## parse img directory
            self.imgs = {}
            imgnames = []
            impth = './data/Cityscapes/images/val'               
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
            gtpth = './data/Cityscapes/gtFine/val'
            folders = os.listdir(gtpth)
            for fd in folders:
                fdpth = osp.join(gtpth, fd)
                lbnames = os.listdir(fdpth)
                lbnames = [el for el in lbnames if 'labelTrainIds' in el]
                names = [el.replace('_gtFine_labelTrainIds.png', '') for el in lbnames]
                lbpths = [osp.join(fdpth, el) for el in lbnames]
                gtnames.extend(names)
                self.labels.update(dict(zip(names, lbpths)))
                    
        else:         
            ## parse img directory
            self.imgs = {}
            imgnames = []
            impth = './data/GTA5/images/val'
            folders = os.listdir(impth)
            for fd in folders:
                fdpth = osp.join(impth, fd)
                im_names = os.listdir(fdpth)
                names = [el.replace('.png', '') for el in im_names]
                impths = [osp.join(fdpth, el) for el in im_names]
                imgnames.extend(names)
                self.imgs.update(dict(zip(names, impths)))

            ## parse gt directory
            self.labels = {}
            gtnames = []        
            gtpth = './data/GTA5/labels__/val'
            folders = os.listdir(gtpth)
            for fd in folders:
                fdpth = osp.join(gtpth, fd)
                lbnames = os.listdir(fdpth)
                lbnames = [el for el in lbnames if 'labelTrainIds' in el]
                names = [el.replace('_labelTrainIds.png', '') for el in lbnames]
                lbpths = [osp.join(fdpth, el) for el in lbnames]
                gtnames.extend(names)
                self.labels.update(dict(zip(names, lbpths)))

        self.imnames = imgnames[random_index:random_index+2]
        self.len = len(self.imnames)

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
    
    
    def __getitem__(self, index):
        fn  = self.imnames[index]
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
    
def testimg(test_set,weight_pth):
    args = parse_args()

    ## dataset
    n_classes = args.num_classes

    mode = args.mode
        
    # load original image
    imgpths = []
    if test_set=='cityscapes':
        impth = './data/Cityscapes/images/val'
       
    else:
        impth = './data/GTA5/images/val'

    folders = os.listdir(impth)
    for fd in folders:
        fdpth = osp.join(impth, fd)
        im_names = os.listdir(fdpth)
        impths = [osp.join(fdpth, el) for el in im_names]
        imgpths.extend(impths)
                     
    random_index = random.randint(0,len(imgpths) - 1)
    imgpth = imgpths[random_index+1]               
    
    testset = testdata(test_set,random_index)
    dataloader_test = DataLoader( testset,
                       batch_size=1,
                       shuffle=False,
                       num_workers=1,
                       drop_last=False)
                         
    precision, miou, predict, label = test(dataloader_test,weight_pth)
        
    return  precision, miou, predict, label, imgpth
                              
    
def viwer(test_set_ct,test_set_gta, weight_pth_ct,weight_pth_gta):
    
    # cituscapes
    precision_ct, miou_ct, predict_ct, label_ct,imgpth_ct = testimg(test_set_ct,weight_pth_gta)
    
    ori_img_ct = Image.open(imgpth_ct)
    gt_img_ct = gray_to_rgb(label_ct)
    pre_img_ct = gray_to_rgb(predict_ct)
    
    # GTA5
    # precision_gta, miou_gta, predict_gta, label_gta,imgpth_gta = testimg(test_set_gta,weight_pth_gta)
    
    # ori_img_gta = Image.open(imgpth_gta)
    # gt_img_gta = gray_to_rgb(label_gta)
    # pre_img_gta = gray_to_rgb(predict_gta)
    
    fig, axs = plt.subplots(2, 3)   
    axs[0,0].imshow(ori_img_ct)
    axs[0,0].set_title('Original')
    axs[0,0].axis('off')
    
    axs[0,1].imshow(gt_img_ct)
    axs[0,1].set_title('Ground Truth')
    axs[0,1].axis('off')
    
    axs[0,2].imshow(pre_img_ct)
    axs[0,2].set_title('Predict')
    axs[0,2].axis('off')
    
    # axs[1,0].imshow(ori_img_gta)
    # axs[1,0].axis('off')
    # axs[1,1].imshow(gt_img_gta)
    # axs[1,1].axis('off')
    # axs[1,2].imshow(pre_img_gta)
    # axs[1,2].axis('off')
    
    plt.subplots_adjust(wspace=0,hspace=0)
    plt.show()           
    

if __name__ == "__main__":
    # dataset to be tested
    test_set_ct = 'cityscapes'
    test_set_gta = 'GTA5'
    # weights path to be loaded
    weight_pth_ct = './bestweights/cityscapes/best.pth'
    # weight_pth_gta = './bestweights/gta5/aug/best.pth'
    weight_pth_gta = './bestweights/gta5/fda/best_aug.pth'
    
    viwer(test_set_ct,test_set_gta, weight_pth_ct,weight_pth_gta)
    
    


    