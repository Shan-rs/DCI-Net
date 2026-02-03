#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from MyModel import DynamicDecoder
import os
#os.environ["CUDA_LAUNCH_BLOCKING"] = "1"

from torch.utils.data import DataLoader
from torch.utils.data.sampler import SubsetRandomSampler
import torch.nn as nn
from torch import optim
import torchvision.utils
import torch
from config import Config
from DehazingSet import DehazingSet
import visdom
from utils import save_load as sl
from torchmetrics.image import PeakSignalNoiseRatio, StructuralSimilarityIndexMeasure
import utils.MyLoss as MyLoss
import numpy as np
import cv2
import time
import torch.nn.functional as F
from torchvision import transforms as T
from math import log10
from torch.cuda.amp import autocast, GradScaler  # Mixed precision training
from tqdm import tqdm
#from torch.optim.lr_scheduler import CosineAnnealingLR

def get_rotmat(theta):
    theta = torch.tensor(theta).float()
    return torch.tensor([[torch.cos(theta), -torch.sin(theta), 0.],
                          [torch.sin(theta), torch.cos(theta),0.]])
    
def rot_transform(img, theta):
    rot_mat = get_rotmat(theta)[None, ...].cuda().repeat(img.shape[0],1,1)
    grid = F.affine_grid(rot_mat, img.size()).cuda()
    flip_rot = F.grid_sample(img, grid)
    return flip_rot

def train(opt, vis):
#    torch.cuda.set_device(args.dev)
    # Step 1: Initialize model
    model = DynamicDecoder(36)
    
    if torch.cuda.is_available():
        model = model.cuda()
    
    train_sets = DehazingSet(opt.train_data_root, True, True)
    sampler_train = SubsetRandomSampler(torch.randperm(opt.train_num))
    val_sets = DehazingSet(opt.val_data_root, False, False)

    train_dataloader = DataLoader(dataset=train_sets, sampler = sampler_train, batch_size=opt.batch_size, num_workers=opt.num_workers, pin_memory=True, drop_last = True)
    val_dataloader = DataLoader(dataset=val_sets, batch_size=opt.val_batch_size, num_workers=opt.num_workers, pin_memory=True, shuffle=False)

    # Step 3: Loss function and Optimizer
    criterion = nn.L1Loss().cuda()
    criterionSSIM = ssim.SSIM(window_size=11)
    criterionVGG = MyLoss.VGGLoss()
    optimizer = optim.Adam(model.parameters(), lr = opt.lr, betas = (0.9, 0.999), eps=1e-08)#weight_decay = opt.weight_decay
#    scheduler = CosineAnnealingLR(optimizer,T_max=60)

    if opt.load_model_path:
        model, optimizer, epoch_s, step_s = sl.load_state(opt.load_model_path, model, optimizer)
        for param_group in optimizer.param_groups:
            param_group['lr'] = opt.new_lr
    
    if not os.path.exists(opt.output_sample):
        os.mkdir(opt.output_sample)
    
    # metrics
    total_loss = 0
    previous_loss = 2
    max_psnr=0
    max_ssim=0
    """
    Training.
    """
    #This line may increase the training speed a bit.
    torch.backends.cudnn.benchmark = True
    
    # Initialize GradScaler for mixed precision training
    scaler = GradScaler()

    (global_step, step) = (epoch_s + 1, step_s) if opt.load_model_path is not None else (0, 0)
    t0 = time.time()
    
    # Step 5: Start training
    for epoch in range(global_step, opt.max_epoch):
        total_loss = 0
        
        train_bar = tqdm(train_dataloader, desc=f'Epoch {epoch + 1}/{opt.max_epoch}')
        for iteration, (img1, gt1) in enumerate(train_bar):
            model.train()  # train mode
            if torch.cuda.is_available():
                img1 = img1.cuda()
                gt1 = gt1.cuda()
            scale_factor = 0.5
            
            # Use mixed precision training
            with autocast():
                output1, output2, output3, out1 = model(img1)
                
                gt1_1 = F.interpolate(gt1, size=output1.size()[2:], mode='bilinear', align_corners=True)
                gt1_2 = F.interpolate(gt1, size=output2.size()[2:], mode='bilinear', align_corners=True)
                gt1_3 = F.interpolate(gt1, size=output3.size()[2:], mode='bilinear', align_corners=True)
                
                loss_l1 = criterion(out1, gt1) + 0.2*criterion(output1, gt1_1) + 0.4*criterion(output2, gt1_2) + 0.8*criterion(output3, gt1_3)
                loss_l = loss_l1
                loss_SSIM1 = criterionSSIM(out1, gt1)#
                loss_VGG1 = criterionVGG(out1, gt1)
                loss_SSIM = 1-loss_SSIM1
                loss_VGG = loss_VGG1
                loss_co = 0
                if (step + 1) % 5 == 0:  # Calculate collaborative loss every 5 steps for speedup
                    img2 = F.interpolate(img1, scale_factor=scale_factor, mode='bilinear', align_corners=True)  # size=[128, 128]
                    angles = [0, np.pi/2, np.pi, 3*np.pi/2]
                    rot_angle = angles[np.random.choice(4)]
                    img2 = rot_transform(img2, rot_angle)
        #            print(img2.size())
                    gt2 = F.interpolate(gt1,scale_factor=scale_factor,mode='bilinear',align_corners=True)
                    gt2 = rot_transform(gt2, rot_angle)
                    N, C, H, W = img1.size()
                    
                    out1_v2 = F.interpolate(out1,scale_factor=scale_factor,mode='bilinear',align_corners=True)
                    out1_v2 = rot_transform(out1_v2, rot_angle)
                    
                    output21, output22, output23, out2 = model(img2)
                    
                    gt2_1 = F.interpolate(gt2, size=output21.size()[2:], mode='bilinear', align_corners=True)
                    gt2_2 = F.interpolate(gt2, size=output22.size()[2:], mode='bilinear', align_corners=True)
                    gt2_3 = F.interpolate(gt2, size=output23.size()[2:], mode='bilinear', align_corners=True)
                    
                    loss_l2 = criterion(out2, gt2) + 0.2*criterion(output21, gt2_1) + 0.4*criterion(output22, gt2_2) + 0.8*criterion(output23, gt2_3)
                    loss_SSIM2 = criterionSSIM(out2, gt2)#
                    loss_VGG2 = criterionVGG(out2, gt2)
                    loss_SSIM = ((1-loss_SSIM1) + (1-loss_SSIM2))/2
                    loss_VGG = (loss_VGG1 + loss_VGG2)/2
                    loss_co = nn.MSELoss().cuda()(out1_v2, out2)
                    loss_l = (loss_l1 + loss_l2)/2
                    
                loss = loss_co + loss_l + loss_SSIM + loss_VGG
            
            optimizer.zero_grad()
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            total_loss = total_loss + loss.detach()

            # Update progress bar
            train_bar.set_postfix({
                'loss': f'{loss.item():.4f}',
                'l1': f'{loss_l.item():.4f}',
                'co': f'{loss_co.item() if isinstance(loss_co, torch.Tensor) else loss_co:.4f}'
            })

            step = step + 1

            if step % opt.display_iter == 0:
                # visdom logging
                vis.line(X=torch.tensor([step]), Y=torch.tensor([loss]), win='step train based_loss', update='append',
                         name='traning loss')
                vis.line(X=torch.tensor([step]), Y=torch.tensor([loss_co]), win='step train based_loss', update='append',
                         name='Collaborative loss')
                vis.line(X=torch.tensor([step]), Y=torch.tensor([loss_l]), win='step train based_loss', update='append',
                         name='l1 loss')
                vis.line(X=torch.tensor([step]), Y=torch.tensor([loss_SSIM]), win='step train based_loss',
                         update='append', name='ssim_loss')
                vis.line(X=torch.tensor([step]), Y=torch.tensor([loss_VGG]), win='step train based_loss', update='append',
                         name='vgg_loss')
                
#            if step % opt.sample_iter == 0:
#                torchvision.utils.save_image(torch.cat((img1, gt1, out1), dim = 0), \
#                                             opt.output_sample + '/epoch{}_iteration{}_1.jpg'.format(epoch+1, iteration + 1), nrow = opt.batch_size)
#                torchvision.utils.save_image(torch.cat((img2, gt2, out2), dim = 0), \
#                                             opt.output_sample + '/epoch{}_iteration{}_2.jpg'.format(epoch+1, iteration + 1), nrow = opt.batch_size)

            if os.path.exists(opt.debug_file):
                import ipdb
                ipdb.set_trace()
#        scheduler.step()
#            
        training_loss = total_loss / (opt.train_num // opt.batch_size)
        psnr_avg, ssim_avg = val(model, val_dataloader, opt, epoch)

        vis.line(X=torch.tensor([global_step + 1]), Y=torch.tensor([training_loss]), win="loss", update='append',
                 name='train')
        vis.line(X=torch.tensor([global_step + 1]), Y=torch.tensor([psnr_avg]), win="PSNR", update='append', name='val')
        vis.line(X=torch.tensor([global_step + 1]), Y=torch.tensor([ssim_avg]), win="SSIM", update='append', name='psnr')
        print(f'\n epoch:{epoch + 1}| psnr_avg:{psnr_avg:.4f}|ssim_avg:{ssim_avg:.4f}')
        if psnr_avg > max_psnr and ssim_avg > max_ssim:
            max_psnr = max(max_psnr, psnr_avg)
            max_ssim = max(max_ssim, ssim_avg)
            sl.save_state(1, epoch, step, model.state_dict(), optimizer.state_dict())
            print(f'\n model saved at step :{epoch + 1}| max_psnr:{max_psnr:.4f}|max_ssim:{max_ssim:.4f}')

        global_step = global_step + 1
        duration = time.time() - t0
        print("Epoch {}:\tin {} min {:1.2f} sec".format(epoch + 1, duration // 60, duration % 60))
        #if loss does not decrease, decrease learning rate
        if training_loss >= previous_loss:
            for param_group in optimizer.param_groups:
                param_group['lr'] = param_group['lr'] * opt.lr_decay
                sl.save_lr(1, param_group['lr'], epoch, step)
                
        previous_loss = training_loss
        
def val(model, dataloader, opt, epoch):
    model.eval() #evaluation mode
    criterion = nn.L1Loss().cuda()#
    criterion2 = nn.MSELoss().cuda()

    psnr_metric = PeakSignalNoiseRatio(data_range=1.0).cuda()
    ssim_metric = StructuralSimilarityIndexMeasure(data_range=1.0).cuda()
    
    criterionVGG = MyLoss.VGGLoss()
    loss_total = 0
    loss_total_l1 = 0
    psnr_total = 0
    ssim_total = 0
    img_num = 0
    val_bar = tqdm(dataloader, desc='Validation', leave=False)
    for iteration, (img1, gt1) in enumerate(val_bar):
#        print(iteration)
        if torch.cuda.is_available():
            img1 = img1.cuda()
            gt1 = gt1.cuda()
#        N, C, H, W = img1.size()

        with torch.no_grad():
#            print(out_img.size())
            output1, output2, output3, out1 = model(img1)
    
            psnr = psnr_metric(out1, gt1)
            ssim1 = ssim_metric(out1, gt1)

        psnr_total = psnr_total + psnr
        ssim_total = ssim_total + ssim1
        img_num = img_num + 1
#        if iteration % opt.result_sample_iter == 0:
#            torchvision.utils.save_image(torch.cat((img1, gt1, out1), dim = 0), \
#                                     opt.dehazing_result + '/epoch{}_iteration{}_1.jpg'.format(epoch+1, iteration+1), nrow = 3)
#                torchvision.utils.save_image(torch.cat((img2, gt2, out2), dim = 0), \
#                                         opt.dehazing_result + '/epoch{}_iteration{}_2.jpg'.format(epoch+1, iteration+1), nrow = opt.val_batch_size)

    psnr_avg = psnr_total / img_num
    ssim_avg = ssim_total / img_num
    model.train()  # back to train mode

    return psnr_avg, ssim_avg

if __name__ == '__main__':
    opt = Config()
    vis = visdom.Visdom(env = 'DCIL_[]')

    train(opt, vis)
