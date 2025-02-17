import os, errno
import torch

import numpy as np
import torch.nn as nn
import torch.optim as optim

from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import DataLoader

from utils.loss import *
from datasets.syntodd import SynTodd
from datasets.clearpose import ClearPose

    

def multiview_collate(batch):
    targets = []
    for ii in range(len(batch[0])):
        targets.append([batch_element[ii] for batch_element in batch])

    stacked_images = torch.stack(targets[0])
    stacked_camera_poses = torch.stack(targets[1])
    stacked_intrinsics = torch.stack(targets[2])

    return stacked_images, stacked_camera_poses, targets[2], targets[3], targets[4], targets[5], targets[
        6], targets[9]



def get_dataloader(config, mode):
    dataset_name = config["Dataset"]["dataset_name"]
    dataset_config = config["Dataset"][dataset_name]

    if dataset_name == "syntodd":
        dataset = SynTodd(dataset_config, mode)
    elif (dataset_name == "clearpose"):
        dataset = ClearPose(dataset_config, mode)
    else:
        raise NotImplementedError(f'Invalid dataset type: {dataset_name}.')
    
    batch_size = dataset_config["batch_size"]
    num_workers = dataset_config["num_workers"]
    shuffle = dataset_config[f"{mode}_shuffle"]
    
    return DataLoader(
            dataset,
            batch_size=batch_size,
            num_workers=num_workers,
            pin_memory=True,
            shuffle=shuffle,
            drop_last=True
            )


def get_losses(config):
    def NoneFunction(a, b):
        return 0
    loss_depth = NoneFunction
    loss_segmentation = NoneFunction
    loss_depth_grad = NoneFunction
    loss_depth_normal = NoneFunction
    loss_seg_iou = NoneFunction
    loss_seg_error = NoneFunction
    loss_depth_error = NoneFunction
    loss_depth_type = config['Trainer']['loss_depth']
    loss_seg_type = config['Trainer']['loss_segmentation']
    type = config['Model']['type']
    dataset_name = config["Dataset"]["dataset_name"]
    num_classes = len(config['Dataset'][dataset_name]['classes'])
    if 'depth' in type:
        if 'mse' in loss_depth_type:
            # loss_depth = MaskedL2Loss()
            loss_depth = nn.MSELoss()
        elif 'l1' in loss_depth_type:
            loss_depth = MaskedL1Loss()
        elif 'smooth' in loss_depth_type:
            loss_depth = MaskedSmoothL1Loss()
        if 'grad' in loss_depth_type:
            loss_depth_grad = GradientLoss()
        if 'normal' in loss_depth_type:
            # loss_depth_normal = NormalLoss()
            loss_depth_normal = nn.L1Loss()
        
    if 'seg' in type:
        if 'ce' in loss_seg_type:
            loss_segmentation = nn.CrossEntropyLoss()
        if 'iou' in loss_seg_type:
            loss_seg_iou = SegIouLoss(num_classes)
        
    return loss_depth, loss_depth_grad, loss_depth_normal, loss_segmentation, loss_seg_iou


def create_dir(directory):
    try:
        os.makedirs(directory)
    except OSError as e:
        if e.errno != errno.EEXIST:
            raise


def get_optimizer(config, net):
    names = set([name.split('.')[0] for name, _ in net.named_modules()]) - set(['', 'transformer_encoders'])
    params_backbone = net.transformer_encoders.parameters()
    params_scratch = list()
    for name in names:
        params_scratch += list(eval("net."+name).parameters())

    if config['Trainer']['optim'] == 'adam':
        optimizer_backbone = optim.Adam(params_backbone, lr=config['Trainer']['lr_backbone'])
        optimizer_scratch = optim.Adam(params_scratch, lr=config['Trainer']['lr_scratch'])
    elif config['Trainer']['optim'] == 'sgd':
        optimizer_backbone = optim.SGD(params_backbone, lr=config['Trainer']['lr_backbone'], momentum=config['Trainer']['momentum'])
        optimizer_scratch = optim.SGD(params_scratch, lr=config['Trainer']['lr_scratch'], momentum=config['Trainer']['momentum'])
    return optimizer_backbone, optimizer_scratch


def get_schedulers(optimizers):
    return [ReduceLROnPlateau(optimizer) for optimizer in optimizers]
