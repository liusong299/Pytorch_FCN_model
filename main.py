#Basic Imports
#=======================================================================
import logging
import random as rd
import argparse
import resource
import click
import numpy as np
import time
from os.path import abspath, dirname, isdir, isfile, join
import copy
import pickle

#Pytorch Imports
#=======================================================================
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
from torch.autograd import Variable
from torch.utils.data import DataLoader
from torchvision import datasets
from tqdm import tqdm
from distutils.version import LooseVersion
import torch.nn.functional as F
#Self Imports
#=======================================================================
#from utils import *
#from train import *
from dataset import SegDataset
from nets import models

# utils functions
#=======================================================================


def mkdir(newdir):
    if type(newdir) is not str:
        newdir = str(newdir)
    if os.path.isdir(newdir):
        pass
    elif os.path.isfile(newdir):
        raise OSError("a file with the same name as the desired " \
                      "dir, '%s', already exists." % newdir)
    else:
        head, tail = os.path.split(newdir)
        if head and not os.path.isdir(head):
            mkdir(head)
        if tail:
            os.mkdir(newdir)

def get_model_path(name, epoch):
    mkdir('{}/{}'.format(MODEL_DIR, name))
    return '{}/{}/{}'.format(MODEL_DIR, name, epoch)

#train functions
#=======================================================================
def cross_entropy2d(input, target, weight=None, size_average=True):
    '''
    Reference: https://github.com/wkentaro/pytorch-fcn/blob/master/torchfcn/trainer.py
    '''
    # input: (n, c, h, w), target: (n, h, w)
    n, c, h, w = input.size()
    # log_p: (n, c, h, w)
    if LooseVersion(torch.__version__) < LooseVersion('0.3'):
        # ==0.2.X
        log_p = F.log_softmax(input)
    else:
        # >=0.3
        log_p = F.log_softmax(input, dim=1)
    # log_p: (n*h*w, c)
    log_p = log_p.transpose(1, 2).transpose(2, 3).contiguous()
    log_p = log_p[target.view(n, h, w, 1).repeat(1, 1, 1, c) >= 0]
    log_p = log_p.view(-1, c)
    # target: (n*h*w,)
    mask = target >= 0
    target = target[mask]
    loss = F.nll_loss(log_p, target, weight=weight, reduction='sum')
    if size_average:
        loss /= mask.data.sum().float()
    return loss


def fine_tune(model, name):
    logging.info("Fine tuning model: {}".format(name))
    # criterion = nn.CrossEntropyLoss()
    criterion = cross_entropy2d
    optimizer = optim.RMSprop(model.parameters(), lr=lr, momentum=momentum, weight_decay=w_decay)
    scheduler = lr_scheduler.StepLR(optimizer, step_size=step_size,
                                    gamma=gamma)  # decay LR by a factor of 0.5 every 30 epochs

    # TODO data loader
    dsets = {x: SegDataset(os.path.join(DATA_DIR, x)) for x in ['train', 'val']}
    dset_loaders = {x: DataLoader(dsets[x], batch_size=batch_size, shuffle=True, num_workers=1) for x in
                    ['train', 'val']}

    train_loader, val_loader = dset_loaders['train'], dset_loaders['val']
    train(model, name, criterion, optimizer, scheduler, train_loader, val_loader, epochs)


'''
    Reference: https://github.com/pochih/FCN-pytorch/blob/master/python/train.py
'''


def train(model, name, criterion, optimizer, scheduler, train_loader, val_loader, epochs):
    if vals['use_gpu']:
        model = model.cuda()

    for epoch in range(epochs):
        scheduler.step()

        epoch_losses = []
        since = now()
        for iter, batch in enumerate(train_loader):
            optimizer.zero_grad()

            raw_inputs, raw_labels = batch[0], batch[1]

            # inputs, labels = None, None
            if vals['use_gpu']:
                inputs = Variable(raw_inputs.cuda())
                labels = Variable(raw_labels.cuda())
            else:
                inputs, labels = Variable(raw_inputs), Variable(raw_labels)

            outputs = model(inputs)
            # print('Shape. input:{}; output:{}; label:{}'.format(inputs.shape, outputs.shape, labels.shape))

            # loss = criterion(outputs.squeeze(0), labels.squeeze(0))
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            epoch_losses.append(loss.item())
            if iter % 10 == 0:
                logging.info("epoch: {}, iter: {}, loss: {:0.5f}, avg: {:0.5f}"
                             .format(epoch, iter, loss.item(), np.mean(epoch_losses)))

        logging.info("Finish epoch: {}, time: {}, avg_loss: {:0.5f}"
                     .format(epoch, gap_time(since), np.mean(epoch_losses)))
        torch.save(model, get_model_path(name, epoch))

        val(model, val_loader, epoch)


def val(model, val_loader, epoch):
    model.eval()
    total_ious = []
    pixel_accs = []
    for iter, batch in enumerate(val_loader):
        # print('val : {}'.format(len(batch)))
        raw_inputs, raw_labels = batch[0], batch[1]
        if gpu_id >= 0:
            inputs = Variable(raw_inputs.cuda())
        else:
            inputs = Variable(raw_inputs)

        output = model(inputs)
        output = output.data.cpu().numpy()

        N, _, h, w = output.shape
        pred = output.transpose(0, 2, 3, 1).reshape(-1, n_class).argmax(axis=1).reshape(N, h, w)

        target = batch[1].cpu().numpy().reshape(N, h, w)
        for p, t in zip(pred, target):
            total_ious.append(iou(p, t))
            pixel_accs.append(pixel_acc(p, t))

    # Calculate average IoU
    total_ious = np.array(total_ious).T  # n_class * val_len
    ious = np.nanmean(total_ious, axis=1)
    pixel_accs = np.array(pixel_accs).mean()
    logging.info("epoch: {}, pix_acc: {}, meanIoU: {}, IoUs: {}".format(epoch, pixel_accs, np.nanmean(ious), ious))

    # IU_scores[epoch] = ious
    # np.save(os.path.join(score_dir, "meanIU"), IU_scores)
    # pixel_scores[epoch] = pixel_accs
    # np.save(os.path.join(score_dir, "meanPixel"), pixel_scores)


# Calculates class intersections over unions
def iou(pred, target):
    ious = []
    for cls in range(n_class):
        pred_inds = pred == cls
        target_inds = target == cls
        intersection = pred_inds[target_inds].sum()
        union = pred_inds.sum() + target_inds.sum() - intersection
        if union == 0:
            ious.append(float('nan'))  # if there is no ground truth, do not include in evaluation
        else:
            ious.append(float(intersection) / max(union, 1))
        # print("cls", cls, pred_inds.sum(), target_inds.sum(), intersection, float(intersection) / max(union, 1))
    return ious


def pixel_acc(pred, target):
    correct = (pred == target).sum()
    total = (target == target).sum()



rlimit = resource.getrlimit(resource.RLIMIT_NOFILE)
resource.setrlimit(resource.RLIMIT_NOFILE, (1000, rlimit[1]))

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s %(levelname)s %(message)s'
)


import configparser
import sys,os
config = configparser.ConfigParser()
config.read('config.ini')
print(config.sections())
n_class    = config.getint(  'Default', 'n_class')
batch_size = config.getint(  'Default', 'batch_size')
epochs     = config.getint(  'Default', 'epochs')
lr         = config.getfloat('Default', 'lr')
momentum   = config.getint(  'Default', 'momentum')
w_decay    = config.getfloat('Default', 'w_decay')
step_size  = config.getint(  'Default', 'step_size')
gamma      = config.getfloat('Default', 'gamma')
model      = config.get(     'Default', 'model')
gpu_id     = config.getint(  'Default', 'gpu_id')
print(config.items('Default'))
# dir path
ROOT_DIR = dirname(abspath(__file__))
MODEL_DIR = '{}/saved_model'.format(ROOT_DIR)
DATA_DIR = '{}/data'.format(ROOT_DIR)

print("ROOT_DIR=", ROOT_DIR)
print("MODEL_DIR=", MODEL_DIR)
print("DATA_DIR=", DATA_DIR)

if gpu_id >= 0 and torch.cuda.is_available():
    #vals['use_gpu'] = 1
    logging.info('Use GPU. device: {}'.format(gpu_id))
    torch.cuda.set_device(gpu_id)

model = models.all_models[model](n_class)
fine_tune(model, model)

# TODO calculate mean of BGR
means = np.array([104.00698793, 116.66876762, 122.67891434])



now = lambda: time.time()
gap_time = lambda past_time : int((now() - past_time) * 1000)



