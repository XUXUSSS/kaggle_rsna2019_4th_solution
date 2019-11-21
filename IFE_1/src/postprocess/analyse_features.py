
import os
import argparse
import logging
import pickle

import numpy as np
import shutil

from multiprocessing import Pool
from math import log as ln
from IPython import embed
from src.cnn.utils.config import Config

parser = argparse.ArgumentParser()
parser.add_argument('config')
parser.add_argument("--pkl", default='/mnt/WXRC0020/users/mdxu/08_RSNA19/02_post_exps/L01/model/model001/fold0_ep23_test_tta10.pkl', type=str, help='result pickle file')
parser.add_argument('--output', default='./test')
parser.add_argument('--fold',default=0)
parser.add_argument('--istest',default=False)
parser.add_argument('--ttaid',default=0)
parser.add_argument('--datafold',default=0)

args = parser.parse_args()


def process(input):

    id = input[0]
    target = input[1]
    output = input[2]
    feature = input[3]
    feature_3D = input[4]
    if args.istest:
        dst = os.path.join(args.output , 'fold{}_test_{}/{}.npy'.format(args.fold,args.ttaid,id))
    else:
        dst = os.path.join(args.output , 'fold{}_train/{}.npy'.format(args.fold,id))

    features={}
    preds=np.clip(output,1e-15,1-1e-15)
    features['pred'] = preds
    features['target'] = target
    features['feature'] = feature
    features['feature_3D'] = feature_3D
    np.save(dst,features)


def main():
    cfg = Config.fromfile(args.config)
    #load features
    print(args.pkl)
    f = open(args.pkl, 'rb')
    #shutil.rmtree(args.output)

    if args.istest:
        output = os.path.join(args.output, 'fold{}_test_{}'.format(args.fold,args.ttaid))
        annotations_pkl  = cfg.data.test.annotations
    else:
        output = os.path.join(args.output, 'fold{}_train'.format(args.fold))
        annotations_pkl = cfg.data.train.annotations

    if not os.path.exists(output):
        os.mkdir(output)
    ret = pickle.load(f)[int(args.ttaid)]

    # group annotations
    with open(annotations_pkl, 'rb') as f:
        df = pickle.load(f)
    if not args.istest:
        df = df[df.fold.isin([args.datafold])]

    df_StudyCount = df.groupby('PatientID')['StudyInstanceUID'].apply(lambda x: len(list(np.unique(x)))).rename('StudyCount')
    df = df.merge(df_StudyCount, left_on='PatientID', right_on = 'PatientID')

    df_sorted_grouped = df.sort_values(['SeriesInstanceUID', 'Position3']).groupby('SeriesInstanceUID')

    df_Thickness = df_sorted_grouped['Position3'].apply(lambda x: x.diff().mean()).rename('Thickness')
    df = df.merge(df_Thickness, left_on='SeriesInstanceUID', right_on = 'SeriesInstanceUID')

    df_FirstZ = df_sorted_grouped['Position3'].first().rename('FirstZ')
    df = df.merge(df_FirstZ, left_on='SeriesInstanceUID', right_on = 'SeriesInstanceUID')
    df['DistanceZ'] = df['Position3'] - df['FirstZ']

    df = df.sort_values(['SeriesInstanceUID','Position3']).groupby(['SeriesInstanceUID'])['ID', 'labels', 'Thickness', 'StudyCount', 'DistanceZ'].agg(lambda x: ','.join(x.dropna().astype(str)))
    print('finish initialization')

    imgs = {}
    #embed()
    for i in range(len(ret['ids'])):

        imgid = ret['ids'][i]
        res = {}
        res['pred'] = ret['outputs'][i]
        res['feature'] = ret['features'][i]
        res['feature_3D'] = ret['features_3D'][i]
        res['target'] = ret['targets'][i]

        imgs[imgid] = res

    for idx in range(len(df)):
        row = df.iloc[idx]
        list_ID = row.ID.split(',')
        if list_ID[0] in imgs:
            image = np.array([[0.0] * (2048 + 6)]*60)
            for i, ID in enumerate(list_ID):
                data = imgs[ID]
                image[i,:] = np.concatenate([data['feature'], data['pred']])

            np.save(output + '/' + list_ID[0] + '.npy', image)



if __name__ == '__main__':
    main()
