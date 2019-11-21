import os
import pickle
import random

import pandas as pd
import numpy as np
import cv2
import torch
import pydicom

from .. import factory
from ..utils.logger import log
from ...utils import mappings, misc


def apply_pixeljitter(data, vars = 0.1, isuniform=True):
    h,w,c = data.shape
    rng = np.random.RandomState()
    if isuniform:
        low = rng.uniform(-vars, 0)
        high = rng.uniform(0, vars)
        random_pixel = rng.uniform(low=low, high=high, size=(h,w,c)).astype(np.float32)
    else:
        random_pixel = rng.normal(0, vars, size=(h,w,c)).astype(np.float32)

    augmented_data = data + random_pixel
    return augmented_data


def apply_channeljitter(data, vars = [0.05, 0.05, 0.05], isuniform=True):
    h,w,c = data.shape
    rng = np.random.RandomState()
    augmented_data = np.zeros(data.shape)
    data_min = data.min()
    data_max = data.max()

    if isuniform:
        random_vars = [rng.uniform(-x, x) for x in vars]
    else:
        random_vars = [rng.normal(0, x) for x in vars]
    for ic in range(c):
        var = random_vars[ic]
        augmented_data[:,:,ic] = np.minimum(np.maximum(data[:,:,ic] + var, data_min), data_max)
    return augmented_data


def apply_window_policy(image, row, policy):
    if policy.index == 1:
        image1 = misc.apply_window(image, 40, 80) # brain
        image2 = misc.apply_window(image, 80, 200) # subdural
        image3 = misc.apply_window(image, row.WindowCenter, row.WindowWidth)
        image1 = (image1 - 0) / 80
        image2 = (image2 - (-20)) / 200
        image3 = (image3 - image3.min()) / (image3.max()-image3.min())
        image = np.array([
            image1 - image1.mean(),
            image2 - image2.mean(),
            image3 - image3.mean(),
        ]).transpose(1,2,0)
    elif policy.index == 2:
#        image1 = misc.apply_window(image, 40, 80) # brain
#        image2 = misc.apply_window(image, 80, 200) # subdural
#        image3 = misc.apply_window(image, 40, 380) # bone
#        image1 = (image1 - 0) / 80
#        image2 = (image2 - (-20)) / 200
#        image3 = (image3 - (-150)) / 380

        wcenter = [40, 80, 40]
        wwidth = [80, 200, 380]
        if policy.noise:
            rng = np.random.RandomState()
            random_vars = []
            if policy.isuniform:
                for row in policy.range:
                    random_vars.append([rng.uniform(-row[i], row[i]) for i in range(len(row))])
            else:
                for row in policy.range:
                    random_vars.append([rng.normal(0, row[i]) for i in range(len(row))])
            wcenter = np.array(wcenter) + np.array(random_vars[0])
            wwidth = np.array(wwidth) + np.array(random_vars[1])

        image1 = misc.apply_window(image, wcenter[0], wwidth[0]) # brain
        image2 = misc.apply_window(image, wcenter[1], wwidth[1]) # subdural
        image3 = misc.apply_window(image, wcenter[2], wwidth[2]) # bone
        image1 = (image1 - (wcenter[0] - wwidth[0]//2)) / wwidth[0]
        image2 = (image2 - (wcenter[1] - wwidth[1]//2)) / wwidth[1]
        image3 = (image3 - (wcenter[2] - wwidth[2]//2)) / wwidth[2]
        image = np.array([
            image1,
            image2,
            image3,
        ]).transpose(1,2,0)

        if policy.drop:
            index = np.random.choice(range(3),1)
            image[:,:, index] = image[:,:,index].min()

    else:
        raise

    return image


def apply_dataset_policy(df, policy):
    if policy == 'all':
        pass
    elif policy == 'pos==neg':
        df_positive = df[df.labels != '']
        df_negative = df[df.labels == '']
        df_sampled = df_negative.sample(len(df_positive))
        df = pd.concat([df_positive, df_sampled], sort=False)
    else:
        raise
    log('applied dataset_policy %s (%d records)' % (policy, len(df)))

    return df


class CustomDataset(torch.utils.data.Dataset):

    def __init__(self, cfg, folds):
        self.cfg = cfg

        log(f'dataset_policy: {self.cfg.dataset_policy}')
        log(f'window_policy: {self.cfg.window_policy}')

        self.transforms = factory.get_transforms(self.cfg)
        with open(cfg.annotations, 'rb') as f:
            self.df = pickle.load(f)

        if folds:
            self.df = self.df[self.df.fold.isin(folds)]
            log('read dataset (%d records)' % len(self.df))

        self.df = apply_dataset_policy(self.df, self.cfg.dataset_policy)
        #self.df = self.df.sample(560)

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]

        path = '%s/%s.dcm' % (self.cfg.imgdir, row.ID)

        dicom = pydicom.dcmread(path)
        image = dicom.pixel_array
        image = misc.rescale_image(image, row.RescaleSlope, row.RescaleIntercept)
        image = apply_window_policy(image, row, self.cfg.window_policy)

        if self.cfg.mytransforms.apply:
            if self.cfg.mytransforms.pixeljitter.apply:
                image = apply_pixeljitter(image, self.cfg.mytransforms.pixeljitter.range, self.cfg.mytransforms.pixeljitter.isuniform)
            if self.cfg.mytransforms.channeljitter.apply:
                image = apply_channeljitter(image, self.cfg.mytransforms.channeljitter.range, self.cfg.mytransforms.channeljitter.isuniform)

        image = self.transforms(image=image)['image']

        target = np.array([0.0] * len(mappings.label_to_num))
        for label in row.labels.split():
            cls = mappings.label_to_num[label]
            target[cls] = 1.0

        if hasattr(self.cfg, 'spread_diagnosis'):
            for label in row.LeftLabel.split() + row.RightLabel.split():
                cls = mappings.label_to_num[label]
                target[cls] += self.cfg.propagate_diagnosis
        target = np.clip(target, 0.0, 1.0)

        return image, torch.FloatTensor(target), row.ID
