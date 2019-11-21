import os
import pickle
import random

import pandas as pd
import numpy as np
import cv2
import torch
import pydicom
from IPython import embed

from .. import factory
from ..utils.logger import log
from ...utils import mappings, misc



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

        self.transforms = factory.get_transforms(self.cfg)
        with open(cfg.annotations, 'rb') as f:
            self.df = pickle.load(f)

        if folds:
            self.df = self.df[self.df.fold.isin(folds)]
            log('read dataset (%d records)' % len(self.df))


        self.df = apply_dataset_policy(self.df, self.cfg.dataset_policy)

        df_StudyCount = self.df.groupby('PatientID')['StudyInstanceUID'].apply(lambda x: len(list(np.unique(x)))).rename('StudyCount')
        self.df = self.df.merge(df_StudyCount, left_on='PatientID', right_on = 'PatientID')

        df_sorted_grouped = self.df.sort_values(['SeriesInstanceUID', 'Position3']).groupby('SeriesInstanceUID')

        df_Thickness = df_sorted_grouped['Position3'].apply(lambda x: x.diff().mean()).rename('Thickness')
        self.df = self.df.merge(df_Thickness, left_on='SeriesInstanceUID', right_on = 'SeriesInstanceUID')

        df_FirstZ = df_sorted_grouped['Position3'].first().rename('FirstZ')
        self.df = self.df.merge(df_FirstZ, left_on='SeriesInstanceUID', right_on = 'SeriesInstanceUID')
        self.df['DistanceZ'] = self.df['Position3'] - self.df['FirstZ']

        self.df = self.df.sort_values(['SeriesInstanceUID','Position3']).groupby(['SeriesInstanceUID'])['ID', 'labels', 'Thickness', 'StudyCount', 'DistanceZ'].agg(lambda x: ','.join(x.dropna().astype(str)))
        print('finish initialization')

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]

        list_labels = row.labels.split(',')
        list_ID = row.ID.split(',')
        list_studycount = row.StudyCount.split(',')
        list_thickness = row.Thickness.split(',')
        list_distancez = row.DistanceZ.split(',')

        target = np.array([[0.0] * len(mappings.label_to_num)]*self.cfg.imgsize[0])
        for i, labels in enumerate(list_labels):
            for label in labels.split(' '):
                if label in mappings.label_to_num:
                    cls = mappings.label_to_num[label]
                    target[i,cls] = 1.0
        target = target.transpose((1,0))
        target = np.clip(target, 0.0, 1.0)

        if self.cfg.ttaid:
            imgdir = self.cfg.imgdir + '_' + str(self.cfg.ttaid)
        else:
            imgdir = self.cfg.imgdir

        #image3d = np.load(imgdir.replace('input','input3d') + '/' + list_ID[0] + '.npy', allow_pickle=True)
        image  = np.load(imgdir+'/'+ list_ID[0] + '.npy', allow_pickle=True)

        fea_combine = np.array([[0.0] * (5+5)]*self.cfg.imgsize[0])
        for i in range(len(list_ID)):
            study_count = np.clip(int(list_studycount[i]),1,5)
            fea_studycount = np.array([0.0] * 5)
            fea_studycount[:int(study_count)] = 1.00

            thickness = np.clip(round(float(list_thickness[i])), 3,7)
            fea_thickness = np.array([0.0] * 5)
            fea_thickness[:(int(thickness)-2)] = 1.0

            distancez = np.clip(round(float(list_distancez[i])/20), 0,10)
            fea_distancez = np.array([0.0] * 11)
            fea_distancez[:int(distancez)] = 1.0

            fea_combine[i,:] = np.concatenate([fea_thickness, fea_studycount])
        image2 = np.concatenate([fea_combine, image], axis=1)

        image2 = self.transforms(image=image2)['image']
        return image2, torch.FloatTensor(target), row.ID
