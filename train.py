import time
import os
import copy
import argparse
import pdb
import collections
import sys

import numpy as np

import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
from torch.autograd import Variable
from torchvision import datasets, models, transforms
import torchvision
from tensorboardX import SummaryWriter

import model_level_attention
from anchors import Anchors
from dataloader import CSVDataset, collater, Resizer, AspectRatioBasedSampler, Augmenter, UnNormalizer, Normalizer
from torch.utils.data import Dataset, DataLoader

import csv_eval
import cv2
assert torch.__version__.split('.')[1] == '4'

print('CUDA available: {}'.format(torch.cuda.is_available()))

ckpt =  False
def main(args=None):

    parser = argparse.ArgumentParser(description='Simple training script for training a RetinaNet network.')

    parser.add_argument('--csv_train', help='Path to file containing training annotations (see readme)')
    parser.add_argument('--csv_classes', help='Path to file containing class list (see readme)')
    parser.add_argument('--csv_val', help='Path to file containing validation annotations (optional, see readme)')

    parser.add_argument('--depth', help='Resnet depth, must be one of 18, 34, 50, 101, 152', type=int, default=50)
    parser.add_argument('--epochs', help='Number of epochs', type=int, default=50)

    parser.add_argument('--model_name', help='name of the model to save')
    parser.add_argument('--pretrained', help='pretrained model name')

    parser = parser.parse_args(args)

    # Create the data loaders
    dataset_train = CSVDataset(train_file=parser.csv_train, class_list=parser.csv_classes, transform=transforms.Compose([Resizer(), Augmenter(), Normalizer()]))

    if parser.csv_val is None:
        dataset_val = None
        print('No validation annotations provided.')
    else:
        dataset_val = CSVDataset(train_file=parser.csv_val, class_list=parser.csv_classes, transform=transforms.Compose([Resizer(), Normalizer()]))

    sampler = AspectRatioBasedSampler(dataset_train, batch_size=2, drop_last=False)
    dataloader_train = DataLoader(dataset_train, num_workers=16, collate_fn=collater, batch_sampler=sampler)
    #dataloader_train = DataLoader(dataset_train, num_workers=16, collate_fn=collater, batch_size=8, shuffle=True)

    if dataset_val is not None:
        sampler_val = AspectRatioBasedSampler(dataset_val, batch_size=2, drop_last=False)
        dataloader_val = DataLoader(dataset_val, num_workers=16, collate_fn=collater, batch_sampler=sampler_val)
        #dataloader_val = DataLoader(dataset_train, num_workers=16, collate_fn=collater, batch_size=8, shuffle=True)

    # Create the model_pose_level_attention
    if parser.depth == 18:
        retinanet = model_level_attention.resnet18(num_classes=dataset_train.num_classes())
    elif parser.depth == 34:
        retinanet = model_level_attention.resnet34(num_classes=dataset_train.num_classes())
    elif parser.depth == 50:
        retinanet = model_level_attention.resnet50(num_classes=dataset_train.num_classes())
    elif parser.depth == 101:
        retinanet = model_level_attention.resnet101(num_classes=dataset_train.num_classes())
    elif parser.depth == 152:
        retinanet = model_level_attention.resnet152(num_classes=dataset_train.num_classes())
    else:
        raise ValueError('Unsupported model depth, must be one of 18, 34, 50, 101, 152')

    if ckpt:
        retinanet = torch.load('')
        print('load ckpt')
    else:
        retinanet_dict = retinanet.state_dict()
        pretrained_dict = torch.load('./weight/' + parser.pretrained)
        pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in retinanet_dict}
        retinanet_dict.update(pretrained_dict)
        retinanet.load_state_dict(retinanet_dict)
        print('load pretrained backbone')

    print(retinanet)
    retinanet = torch.nn.DataParallel(retinanet, device_ids=[0])
    retinanet.cuda()

    retinanet.training = True

    optimizer = optim.Adam(retinanet.parameters(), lr=1e-5)
    #optimizer = optim.SGD(retinanet.parameters(), lr=1e-3, momentum=0.9, weight_decay=1e-4)

    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=3, verbose=True)
    #scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=20, gamma=0.1)

    loss_hist = collections.deque(maxlen=500)

    retinanet.train()
    retinanet.module.freeze_bn()

    print('Num training images: {}'.format(len(dataset_train)))
    f_map = open('./mAP_txt/' + parser.model_name + '.txt', 'a')
    writer = SummaryWriter(log_dir='./summary')
    iters = 0
    for epoch_num in range(0, parser.epochs):

        retinanet.train()
        retinanet.module.freeze_bn()

        epoch_loss = []
        #scheduler.step()

        for iter_num, data in enumerate(dataloader_train):

            iters += 1

            optimizer.zero_grad()

            classification_loss, regression_loss, mask_loss = retinanet([data['img'].cuda().float(), data['annot']])

            classification_loss = classification_loss.mean()
            regression_loss = regression_loss.mean()
            mask_loss = mask_loss.mean()

            loss = classification_loss + regression_loss + mask_loss

            if bool(loss == 0):
                continue

            loss.backward()

            torch.nn.utils.clip_grad_norm_(retinanet.parameters(), 0.1)

            optimizer.step()

            loss_hist.append(float(loss))

            epoch_loss.append(float(loss))

            print('Epoch: {} | Iteration: {} | Classification loss: {:1.5f} | Regression loss: {:1.5f} | mask_loss {:1.5f} | Running loss: {:1.5f}'.format(epoch_num, iter_num, float(classification_loss), float(regression_loss), float(mask_loss), np.mean(loss_hist)))

            writer.add_scalar('classification_loss', classification_loss, iters)
            writer.add_scalar('regression_loss', regression_loss, iters)
            writer.add_scalar('loss', loss, iters)

            del classification_loss
            del regression_loss


        if parser.csv_val is not None:

            print('Evaluating dataset')

            mAP = csv_eval.evaluate(dataset_val, retinanet)
            f_map.write('mAP:{}, epoch:{}'.format(mAP[0][0], epoch_num))
            f_map.write('\n')

        scheduler.step(np.mean(epoch_loss))

        torch.save(retinanet.module, './ckpt/' + parser.model_name + '_{}.pt'.format(epoch_num))

    retinanet.eval()

    writer.export_scalars_to_json("./summary/' + parser.pretrained + 'all_scalars.json")
    f_map.close()
    writer.close()

if __name__ == '__main__':
    main()
