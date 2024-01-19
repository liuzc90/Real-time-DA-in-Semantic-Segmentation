#!/usr/bin/python
# -*- encoding: utf-8 -*-
from model.model_stages import BiSeNet
from gta5 import Gta5
from cityscapes import CityScapes
import torch
from torch.utils.data import DataLoader, SubsetRandomSampler
import logging
import argparse
import numpy as np
from tensorboardX import SummaryWriter
import torch.cuda.amp as amp
from utils import poly_lr_scheduler
from utils import reverse_one_hot, compute_global_accuracy, fast_hist, per_class_iu
from tqdm import tqdm
import os


logger = logging.getLogger()

def is_folder_not_empty(folder_path):
    content = os.listdir(folder_path)
    if len(content) == 0:
        return False
    else:
        return True

def val(args, model, dataloader):
    print('start val!')
    with torch.no_grad():
        model.eval()
        precision_record = []
        hist = np.zeros((args.num_classes, args.num_classes))
        for i, (data, label) in enumerate(dataloader):
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

        return precision, miou


def train(args, model, optimizer, dataloader_train, dataloader_val,load_epoch):
    writer = SummaryWriter(comment=''.format(args.optimizer))

    scaler = amp.GradScaler()
    

    loss_func = torch.nn.CrossEntropyLoss(ignore_index=255)
    max_miou = 0
    step = 0
    for epoch in range(load_epoch,args.num_epochs):
        lr = poly_lr_scheduler(optimizer, args.learning_rate, iter=epoch, max_iter=args.num_epochs)
        model.train()        
        tq = tqdm(total=len(dataloader_train) * args.batch_size)
        tq.set_description('epoch %d, lr %f' % (epoch, lr))
        loss_record = []
        for i, (data, label) in enumerate(dataloader_train):
            data = data.cuda()
            label = label.long().cuda()
            optimizer.zero_grad()

            with amp.autocast():
                output, out16, out32 = model(data)
                loss1 = loss_func(output, label.squeeze(1))
                loss2 = loss_func(out16, label.squeeze(1))
                loss3 = loss_func(out32, label.squeeze(1))
                loss = loss1 + loss2 + loss3

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            tq.update(args.batch_size)
            tq.set_postfix(loss='%.6f' % loss)
            step += 1
            writer.add_scalar('loss_step', loss, step)
            loss_record.append(loss.item())
            
            
        tq.close()
        loss_train_mean = np.mean(loss_record)
        writer.add_scalar('epoch/loss_epoch_train', float(loss_train_mean), epoch)
        print('loss for train : %f' % (loss_train_mean))
        if epoch % args.checkpoint_step == 0 and epoch != 0:
            import os
            if not os.path.isdir(args.save_model_path):
                os.mkdir(args.save_model_path)
            torch.save(model.module.state_dict(), os.path.join(args.save_model_path, 'latest.pth'))


        if epoch % args.validation_step == 0 and epoch != 0:
            precision, miou = val(args, model, dataloader_val)
            
            torch.save({
            'epoch': epoch,
            'model_state_dict': model.module.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            }, './checkpoints/gta5/checkpoint_gta5.tar')
            
            if miou > max_miou:
                max_miou = miou
                import os
                os.makedirs(args.save_model_path, exist_ok=True)
                torch.save(model.module.state_dict(), os.path.join(args.save_model_path, 'best.pth'))
            writer.add_scalar('epoch/precision_val', precision, epoch)
            writer.add_scalar('epoch/miou val', miou, epoch)

def str2bool(v):
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Unsupported value encountered.')

def parse_args():
    # Simulate command-line arguments
    args = argparse.Namespace(
        mode='train',
        backbone='CatmodelSmall',
        pretrain_path='./pretrain/STDCNet813M_73.91.tar',
        use_conv_last=False,
        num_epochs=50,
        epoch_start_i=0,
        checkpoint_step=1,
        validation_step=2,
        crop_height=512,
        crop_width=1024,
        batch_size= 2,
        learning_rate=0.001,
        num_workers=2,
        num_classes=19,
        cuda='0',
        use_gpu=True,
        save_model_path='./checkpoints/gta5/',
        optimizer='adam',
        loss='crossentropy'
    )
    return args

def main(seldata,setaug):
    args = parse_args()

    ## dataset
    n_classes = args.num_classes
       
    if seldata == 'train':
        train_dataset = Gta5(mode=seldata,setaug=setaug)
        dataloader_train = DataLoader(train_dataset,
                        batch_size=args.batch_size,
                        shuffle=False,
                        num_workers=args.num_workers,
                        pin_memory=False,
                        drop_last=True)

        val_dataset = Gta5(mode='val',setaug=setaug)
        dataloader_val = DataLoader(val_dataset,
                           batch_size=1,
                           shuffle=False,
                           num_workers=args.num_workers,
                           drop_last=False)
    if seldata == 'train_fda':          
        train_dataset = Gta5(mode=seldata,setaug=setaug)
        dataloader_train = DataLoader(train_dataset,
                        batch_size=args.batch_size,
                        shuffle=False,
                        num_workers=args.num_workers,
                        pin_memory=False,
                        drop_last=True)
    
        val_dataset = Gta5(mode='val_fda',setaug=setaug)
        dataloader_val = DataLoader(val_dataset,
                           batch_size=1,
                           shuffle=False,
                           num_workers=args.num_workers,
                           drop_last=False)

    ## model
    model = BiSeNet(backbone=args.backbone, n_classes=n_classes, pretrain_model=args.pretrain_path, use_conv_last=args.use_conv_last)

 
    ## optimizer
    # build optimizer
    if args.optimizer == 'rmsprop':
        optimizer = torch.optim.RMSprop(model.parameters(), args.learning_rate)
    elif args.optimizer == 'sgd':
        optimizer = torch.optim.SGD(model.parameters(), args.learning_rate, momentum=0.9, weight_decay=1e-4)
    elif args.optimizer == 'adam':
        optimizer = torch.optim.Adam(model.parameters(), args.learning_rate)
    else:  # rmsprop
        print('not supported optimizer \n')
        return None


    if is_folder_not_empty('./checkpoints/gta5'):
        checkpoint = torch.load('./checkpoints/gta5/checkpoint_gta5.tar')
        model.load_state_dict(checkpoint['model_state_dict'])
        if torch.cuda.is_available() and args.use_gpu:
            model = torch.nn.DataParallel(model).cuda()
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        load_epoch = checkpoint['epoch']+1      
        model.train() # Sets module in training mode
    else:
        if torch.cuda.is_available() and args.use_gpu:
            model = torch.nn.DataParallel(model).cuda()
        load_epoch = 0


    ## train loop
    train(args, model, optimizer, dataloader_train, dataloader_val,load_epoch)
    # final test
    val(args, model, dataloader_val)

if __name__ == "__main__":
    seldata = 'train_fda' # For original data: 'train'; For fda data: 'train_fda'
    setaug = 0 # For data augment:set True; Otherwise: set Fasle
    main(seldata,setaug)