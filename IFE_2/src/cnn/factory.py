import copy

import torch
from torch import nn
from torch.utils.data import DataLoader
import torch.optim
from torch.optim import lr_scheduler
import albumentations as A
from albumentations.pytorch import ToTensor
from efficientnet_pytorch import EfficientNet

from .dataset.custom_dataset import CustomDataset
from .transforms.transforms import RandomResizedCrop
from .utils.logger import log


def get_loss(cfg):
    #loss = getattr(nn, cfg.loss.name)(**cfg.loss.params)
    loss = getattr(nn, cfg.loss.name)(weight=torch.FloatTensor([2,1,1,1,1,1]).cuda(), **cfg.loss.params)
    log('loss: %s' % cfg.loss.name)
    return loss


def get_dataloader(cfg, folds=None):
    dataset = CustomDataset(cfg, folds)
    log('use default(random) sampler')
    loader = DataLoader(dataset, **cfg.loader)
    return loader


def get_transforms(cfg):
    def get_object(transform):
        if hasattr(A, transform.name):
            return getattr(A, transform.name)
        else:
            return eval(transform.name)
    transforms = [get_object(transform)(**transform.params) for transform in cfg.transforms]
    return A.Compose(transforms)


def get_model(cfg):

    log(f'model: {cfg.model.name}')
    log(f'pretrained: {cfg.model.pretrained}')

    if cfg.model.name in ['resnext101_32x8d_wsl']:
        model = torch.hub.load('facebookresearch/WSL-Images', cfg.model.name)
        model.fc = torch.nn.Linear(2048, cfg.model.n_output)
        return model

    if cfg.model.name in ['efficientnetB0']:
        model = EfficientNet.from_pretrained('efficientnet-b0')
        model._fc = torch.nn.Linear(1280,cfg.model.n_output)
        return model




def get_optim(cfg, parameters):
    optim = getattr(torch.optim, cfg.optim.name)(parameters, **cfg.optim.params)
    log(f'optim: {cfg.optim.name}')
    return optim


def get_scheduler(cfg, optim, last_epoch):
    if cfg.scheduler.name == 'ReduceLROnPlateau':
        scheduler = lr_scheduler.ReduceLROnPlateau(
            optim,
            **cfg.scheduler.params,
        )
        scheduler.last_epoch = last_epoch
    else:
        scheduler = getattr(lr_scheduler, cfg.scheduler.name)(
            optim,
            last_epoch=last_epoch,
            **cfg.scheduler.params,
        )
    log(f'last_epoch: {last_epoch}')
    return scheduler

