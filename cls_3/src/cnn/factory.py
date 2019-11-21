import copy

import torch
from torch import nn
from torch.utils.data import DataLoader
import torch.optim
from torch.optim import lr_scheduler
import albumentations as A
from albumentations.pytorch import ToTensor
import torchvision

from .dataset.custom_dataset import CustomDataset
from .transforms.transforms import RandomResizedCrop
from .utils.logger import log
from .models.model import SimpleNet


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


def build_model(cfg):
    model = SimpleNet()
    return model


def get_model(cfg):

    log(f'model: {cfg.model.name}')
    log(f'pretrained: {cfg.model.pretrained}')

    if cfg.model.name in ['resnext101_32x8d_wsl']:
        model = torch.hub.load('facebookresearch/WSL-Images', cfg.model.name)
        model.fc = torch.nn.Linear(2048, cfg.model.n_output)
        if cfg.model.initialbn:
            model = torch.nn.Sequential(torch.nn.BatchNorm2d(num_features=3), model)
        return model

    if cfg.model.name in ['resnet50']:
        model = torchvision.models.resnet50(pretrained=True)
        if cfg.model.dropout is not None:
            model.fc = torch.nn.Sequential(torch.nn.Dropout(cfg.model.dropout), torch.nn.Linear(model.fc.in_features, cfg.model.n_output))
            print(model.fc)
        else:
            model.fc = torch.nn.Linear(model.fc.in_features, cfg.model.n_output)

        if cfg.model.initialbn:
            model = torch.nn.Sequential(torch.nn.BatchNorm2d(num_features=3), model)
        return model

    if cfg.model.name in ['resnext50_32x4d_swsl', 'resnext101_32x4d_swsl','resnext101_32x8d_swsl']:
        model = torch.hub.load('facebookresearch/semi-supervised-ImageNet1K-models', cfg.model.name)
        num_ftrs = model.fc.in_features
        model.fc = torch.nn.Linear(num_ftrs, cfg.model.n_output)
        if cfg.model.initialbn:
            model = torch.nn.Sequential(torch.nn.BatchNorm2d(num_features=3), model)
        return model


    if cfg.model.name in ['efficientnetB0']:
        model = EfficientNet.from_pretrained('efficientnet-b0')
        model._fc = torch.nn.Linear(1280, cfg.model.n_output)
        if cfg.model.initialbn:
            model = torch.nn.Sequential(torch.nn.BatchNorm2d(num_features=3), model)
        return model

    if cfg.model.name in ['efficientnetB4']:
        model = EfficientNet.from_pretrained('efficientnet-b4')
        num_ftrs = model._fc.in_features
        model._fc = torch.nn.Linear(num_ftrs, cfg.model.n_output)
        if cfg.model.initialbn:
            model = torch.nn.Sequential(torch.nn.BatchNorm2d(num_features=3), model)
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

