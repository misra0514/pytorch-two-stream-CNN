#!/usr/bin/env python
# -*- conding: utf-8 -*-

''' 
关于本文件： 感觉是同样是train + test都可以使用的
【注意！！】 使用前记得先  ··· python -m visdom.server ···  启动visdom服务器
    
'''


import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

import torchvision
import torchvision.transforms as transforms
from torchvision.models.resnet import *

from torch.utils.data import DataLoader
from torchvision import datasets

import os
import numpy as np
import argparse
from preprocess import Preprocess
from net import SpatialNet, TemporalNet

import visualize

# args
parser = argparse.ArgumentParser() # 创建一个解析对象
parser.add_argument('--test', dest='test', action='store_true')
parser.set_defaults(test=True)
args = parser.parse_args() # 进行解析

# path
pretrained = 'pretrained'
params_spatial = '/spatial.pth'
params_temporal = '/temporal.pth'

# hyper-params
epochs = 25
batch_size = 4
lr = 0.002
momentum = 0.9

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# pre-processing
# 按照函数执行顺序，在做完了基础的解析器设置、路径和节点设置之后就会进入这里。这里注意需要在preprocess里面填入路径
# 重复执行的话看来一定会经过这个地方，不管test和train都是。不知道能不能注释掉？
Preprocess()

# 신경망 구성 - 神经网络配置
# # 논문의 신경망 - 论文 神经网络配置
#spatialnet = SpatialNet().to(device)
#temporalnet = TemporalNet().to(device)

# resnet101
# 发生异常: RuntimeError       (note: full exception trace is shown but execution is paused at: <module>)
# CUDA error: out of memory
#   File "D:\Programing\python\pytorch-two-stream-CNN\train.py", line 61, in <module>
#     spatialnet = resnet101(num_classes=51).to(device)
#   File "<string>", line 1, in <module> (Current frame)
spatialnet = resnet101(num_classes=51).to(device)
temporalnet = resnet101(num_classes=51).to(device)

print(spatialnet)
print(temporalnet)


# 신경망 파라매터 로드 - 加载神经网络参数
if os.path.isfile(pretrained+params_spatial):
    spatialnet.load_state_dict(torch.load(pretrained+params_spatial))
    temporalnet.load_state_dict(torch.load(pretrained+params_temporal))
    print('\n[*]parameters loaded')


# loss function, optimizer 정의
criterion = nn.CrossEntropyLoss()
optim_rgb = optim.SGD(spatialnet.parameters(), lr=lr, momentum=momentum)
optim_opt = optim.SGD(temporalnet.parameters(), lr=lr, momentum=momentum)
print(optim_rgb)


# 데이터 전처리 정의 - 定义数据预处理
transform = transforms.Compose([
         transforms.Resize(255),
         transforms.RandomCrop(224),
         transforms.ToTensor(),
         transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
         ])


# 데이터 로더 - 数据加载器
dataset = {'{}/{}'.format(x, y) : datasets.ImageFolder(root='data/{}/{}/'.format(x, y),
                                                       transform=transform)
                                                       for x in ['train','test','val']
                                                       for y in ['image', 'optical']}

loader = {'{}/{}'.format(x, y) : DataLoader(dataset=dataset['{}/{}'.format(x, y)],
                                            batch_size=batch_size,
                                            num_workers=2,
                                            shuffle=True)
                                            for x in ['train', 'test', 'val']
                                            for y in ['image', 'optical']}


# 신경망 학습
def train():

    # visdom 그래프 플로팅-图形绘制
    plot_loss = visualize.line('Loss', port=8097)
    plot_loss.register_line('Loss', 'iter', 'loss')
    
    accuracy = {'{}'.format(x) : visualize.line('{}'.format(x), port=8097) for x in ['train/spatial', 'train/temporal', 'val/spatial', 'val/temporal']}
    for x in ['train/spatial', 'train/temporal', 'val/spatial', 'val/temporal']:
        accuracy['{}'.format(x)].register_line('{}'.format(x), 'epoch', 'accuracy')


    train_size = len(dataset['train/image'])
    val_size = len(dataset['val/image'])

    # Training
    for epoch in range(epochs):
        running_loss = 0.0
        correct_rgb = 0
        correct_opt = 0             
        
        # len(loader['train/image']) = 7305，根据batchsize更改
        for i, ((img, rgb_target), (optical, opt_target)) in enumerate(zip(loader['train/image'], loader['train/optical'])):
            print("当前训练进度: ", i ,"/" ,len(loader['train/image']) , " 总进度： ", epoch , " / " , epochs)
            img = img.to(device)
            rgb_target = rgb_target.to(device)

            optical = optical.to(device)
            opt_target = opt_target.to(device)

            optim_rgb.zero_grad()
            optim_opt.zero_grad()

            # 经常在执行以下内同时提示out of memory。所以这里清除下缓存。同时把空间网络去掉
            if hasattr(torch.cuda, 'empty_cache'):
                torch.cuda.empty_cache()
            
            # outputs_spatial = spatialnet(img)
            # pred_spatial = torch.max(outputs_spatial, 1)[1]

            outputs_temporal = temporalnet(optical)
            pred_temporal = torch.max(outputs_temporal, 1)[1]
            
            #loss_spatial = criterion(outputs_spatial, rgb_target)
            loss_temporal = criterion(outputs_temporal, opt_target)
            # loss = loss_spatial + loss_temporal
            loss  =  loss_temporal

            loss.backward()
            optim_rgb.step()
            optim_opt.step()

            running_loss += loss.item()
            # correct_rgb += sum(rgb_target.cpu().numpy() == pred_spatial.cpu().numpy())
            correct_opt += sum(opt_target.cpu().numpy() == pred_temporal.cpu().numpy())


            if (i+1) % 100 == 0:
                print('---Spatial---')
                #print('prediction  : {0} \ntarget      : {1}'.format(pred_spatial, rgb_target))
                print('---Temporal--')
                print('prediction  : {0} \ntarget      : {1}'.format(pred_temporal, opt_target))
                print('[epoch : {0:3d}, {1:5d}/{2}] loss : {3:3f}'.format(epoch+1, (i+1)*batch_size, train_size, running_loss/((i+1)*batch_size)))
                print('') 
        plot_loss.update_line('Loss', epoch, running_loss/train_size)
        accuracy['train/spatial'].update_line('train/spatial', epoch, running_loss/train_size)
        accuracy['train/temporal'].update_line('train/temporal', epoch, running_loss/train_size)
                
        # Validation 每2个epoch做一次验证
        if (epoch+1) % 2 == 0:
            correct_rgb = 0
            correct_opt = 0
            # 验证的时候，会计算一组val里面正确的数量。 eg，在18轮的时候，correct * size / (总数 * batchsize) = 4
            for i, ((img, rgb_target), (optical, opt_target)) in enumerate(zip(loader['val/image'], loader['val/optical'])):
                img = img.to(device)
                rgb_target = rgb_target.to(device)

                optical = optical.to(device)
                opt_target = opt_target.to(device)

                #outputs_spatial = spatialnet(img)
                #pred_spatial = torch.max(outputs_spatial, 1)[1]

                outputs_temporal = temporalnet(img)
                pred_temporal = torch.max(outputs_temporal, 1)[1]

                # correct_rgb += sum(rgb_target.cpu().numpy() == pred_spatial.cpu().numpy())
                correct_opt += sum(opt_target.cpu().numpy() == pred_temporal.cpu().numpy())

            accuracy['val/spatial'].update_line('val/spatial', epoch, correct_rgb/((i+1)*batch_size))
            accuracy['val/temporal'].update_line('val/temporal', epoch, correct_opt/((i+1)*batch_size))
            print('accuracy of spatial      : {}'.format(100*correct_rgb/((i+1)*batch_size)))
            print('accuracy of temporal     : {}'.format(100*correct_opt/((i+1)*batch_size)))
            print('本组正确数量 ： ' +  str(correct_opt))

        # 파라매터 저장- 保存参数
        # 每一个epoch之后都会生成pretrained
        if not os.path.exists(pretrained):
            os.makedirs(pretrained, exist_ok=True)
        torch.save(spatialnet.state_dict(), pretrained+params_spatial+'_{0:03d}'.format(epoch))
        torch.save(temporalnet.state_dict(), pretrained+params_temporal+'_{0:03d}'.format(epoch))


    print('Training Finished')


def test():
    
    # 所以，opt指的是光流的optical。【作为twostrea-flow，包括了光流和一个单个页面的cnn记录内容】所以重点关注optical的正确率
    running_loss = 0.0
    correct_rgb = 0
    correct_opt = 0

    for i, ((img, rgb_target), (optical, opt_target)) in enumerate(zip(loader['test/image'], loader['test/optical'])):
        img = img.to(device)
        rgb_target = rgb_target.to(device)

        optical = optical.to(device)
        opt_target = opt_target.to(device)

        #outputs_spatial = spatialnet(img)
        #pred_spatial = torch.max(outputs_spatial, 1)[1]

        outputs_temporal = temporalnet(optical)
        pred_temporal = torch.max(outputs_temporal, 1)[1]


        #loss_spatial = criterion(outputs_spatial, rgb_target)
        loss_temporal = criterion(outputs_temporal, pred_temporal)
        # loss = loss_spatial + loss_temporal
        loss = loss_temporal

        running_loss += loss.item()
        # correct_rgb += sum(rgb_target.cpu().numpy() == pred_spatial.cpu().numpy())
        correct_opt += sum(opt_target.cpu().numpy() == pred_temporal.cpu().numpy())

        #print('--Spatial--')
        # print('prediction   : {0} \ntarget       : {1} \nloss         : {2}'.format(pred_spatial, rgb_target, running_loss/(i+1)))
        print('--Temporal--')
        print('prediction   : {0} \ntarget       : {1} \nloss         : {2}'.format(pred_temporal, opt_target, running_loss/(i+1)))
        print('当前测试进度: ', i, len(loader['test/optical']))
    print('accuracy of spatial      : {}'.format(100*correct_rgb/((i+1)*batch_size)))
    print('accuracy of temporal     : {}'.format(100*correct_opt/((i+1)*batch_size)))
    print('Test Finished')



# 看来倒是train还是test需要手动调整...args 使用解析器生成的。
# args解析器对象有几个属性。使用add_argument 添加了一个test属性，并且现在是false（可修改）。
def main():
    if args.test:
        test()
    else:
        train()

if __name__ == '__main__':

    main()
