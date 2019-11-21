import sys
import os
import time
import argparse
import random
import collections
import pickle

from apex import amp
import pandas as pd
import numpy as np
from sklearn.metrics import f1_score, roc_auc_score, log_loss

import torch
from torch import nn
import torch.nn.functional as F
from tensorboardX import SummaryWriter

from . import factory
from .utils import util
from .utils.config import Config
from .utils.logger import logger, log
from math import log as ln
from IPython import embed

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('mode', choices=['train', 'valid', 'test'])
    parser.add_argument('config')
    parser.add_argument('--debug', action='store_true')
    parser.add_argument('--fold', type=int, required=True)
    parser.add_argument('--gpu', type=int, default=0)
    parser.add_argument('--snapshot')
    parser.add_argument('--output')
    parser.add_argument('--epoch', type=int, default=30)
    parser.add_argument('--n-tta', default=1, type=int)
    parser.add_argument('--ttaid',default=0)
    return parser.parse_args()



def main():

    args = get_args()
    cfg = Config.fromfile(args.config)

    # copy command line args to cfg
    cfg.mode = args.mode
    cfg.debug = args.debug
    cfg.fold = args.fold
    cfg.snapshot = args.snapshot
    cfg.output = args.output
    cfg.n_tta = args.n_tta
    cfg.gpu = args.gpu
    cfg.data.train.workdir = cfg.workdir
    cfg.data.test.workdir = cfg.workdir
    cfg.data.test.ttaid = args.ttaid
    cfg.data.train.ttaid = False
    cfg.data.valid.ttaid = False
    cfg.data.valid.workdir = cfg.workdir
    cfg.epoch = int(args.epoch)
    cfg.scheduler.params.T_max = int(args.epoch)

    logger.setup(cfg.workdir, name='%s_fold%d' % (cfg.mode, cfg.fold))
    torch.cuda.set_device(cfg.gpu)
    #util.set_seed(cfg.seed)

    log(f'mode: {cfg.mode}')
    log(f'workdir: {cfg.workdir}')
    log(f'fold: {cfg.fold}')
    log(f'batch size: {cfg.batch_size}')
    log(f'acc: {cfg.data.train.n_grad_acc}')
    writer_dict = {
        'writer': SummaryWriter(log_dir='{}/tb_{}_fold{}'.format(cfg.workdir,cfg.mode,cfg.fold)),
        'train_global_steps': 0
    }


    model = factory.build_model(cfg)
    model.cuda()

    if cfg.mode == 'train':
        train(cfg, model, writer_dict = writer_dict)
    elif cfg.mode == 'valid':
        valid(cfg, model)
    elif cfg.mode == 'test':
        test(cfg, model)


def test(cfg, model):
    assert cfg.output
    util.load_model(cfg.snapshot, model)
    loader_test = factory.get_dataloader(cfg.data.test)
    with torch.no_grad():
        results = [run_nn(cfg.data.test, 'test', model, loader_test) for i in range(cfg.n_tta)]
    with open(cfg.output, 'wb') as f:
        pickle.dump(results, f)
    log('saved to %s' % cfg.output)


def valid(cfg, model):
    assert cfg.output
    criterion = factory.get_loss(cfg)
    util.load_model(cfg.snapshot, model)
    loader_valid = factory.get_dataloader(cfg.data.valid, [cfg.fold])
    with torch.no_grad():
        results = [run_nn(cfg.data.valid, 'valid', model, loader_valid, criterion=criterion) for i in range(cfg.n_tta)]
    with open(cfg.output, 'wb') as f:
        pickle.dump(results, f)
    log('saved to %s' % cfg.output)


def train(cfg, model, writer_dict = None):

    criterion = factory.get_loss(cfg)
    #criterion = cutomBCEloss()
    optim = factory.get_optim(cfg, model.parameters())

    best = {
        'loss': float('inf'),
        'score': 0.0,
        'epoch': -1,
    }
    if cfg.resume_from:
        detail = util.load_model(cfg.resume_from, model, optim=optim)
        best.update({
            'loss': detail['loss'],
            'score': detail['score'],
            'epoch': detail['epoch'],
        })
        writer = writer_dict['writer']
        writer_dict['train_global_steps'] = cfg.data.train.epoch_size * detail['epoch']/cfg.data.train.log_size

    folds = [fold for fold in range(cfg.n_fold) if cfg.fold != fold]
    loader_train = factory.get_dataloader(cfg.data.train, folds)
    loader_valid = factory.get_dataloader(cfg.data.valid, [cfg.fold])

    log('train data: loaded %d records' % len(loader_train.dataset))
    log('valid data: loaded %d records' % len(loader_valid.dataset))

    scheduler = factory.get_scheduler(cfg, optim, best['epoch'])

    log('apex %s' % cfg.apex)
    if cfg.apex:
        amp.initialize(model, optim, opt_level='O1')

    for epoch in range(best['epoch']+1, cfg.epoch):

        log(f'\n----- epoch {epoch} -----')

        #util.set_seed(epoch)

        run_nn(cfg.data.train, 'train', model, loader_train, criterion=criterion, optim=optim, apex=cfg.apex, writer_dict = writer_dict)
        with torch.no_grad():
            val = run_nn(cfg.data.valid, 'valid', model, loader_valid, criterion=criterion, writer_dict=writer_dict)

        detail = {
            'score': val['score'],
            'loss': val['loss'],
            'epoch': epoch,
        }
        if val['loss'] <= best['loss']:
            best.update(detail)
            save_path = os.path.join(cfg.workdir,'fold%d_best.pt' % (cfg.fold))
            if (os.path.exists(save_path) and isbetter(save_path, detail)) or (not os.path.exists(save_path)):
                torch.save({
                    'model': model.state_dict(),
                    'optim': optim.state_dict(),
                    'detail': detail,
                }, save_path)
                log('update best epoch model.')

        util.save_model(model, optim, detail, cfg.fold, cfg.workdir)

        log('[best] ep:%d loss:%.4f score:%.4f' % (best['epoch'], best['loss'], best['score']))

        #scheduler.step(val['loss']) # reducelronplateau
        scheduler.step()

def isbetter(path, detail_new):
    state = torch.load(str(path), map_location=lambda storage, location: storage)
    detail_old = state['detail']
    if detail_new['loss'] < detail_old['loss'] :
        return True
    else:
        return False


def run_nn(cfg, mode, model, loader, criterion=None, optim=None, scheduler=None, apex=None, writer_dict=None):
    if mode in ['train']:
        model.train()
    elif mode in ['valid', 'test']:
        model.eval()
    else:
        raise

    t1 = time.time()
    losses = []
    ids_all = []
    targets_all = []
    outputs_all = []

    for i, (inputs, targets, ids) in enumerate(loader):

        batch_size = len(inputs)

        inputs_ = inputs.cuda()
        targets_ = targets.cuda()
        outputs_ = model(inputs_)

        list_id = ids[0].split(',')
        targets=targets_[0,:,:len(list_id)]
        outputs=outputs_[0,:,:len(list_id)]
        inputs=inputs_[0,:,:len(list_id)]
        #embed()
        ids_all.extend(list_id)
        for batch_id in range(1,len(ids)):
            list_id = ids[batch_id].split(',')
            ids_all.extend(list_id)
            targets=torch.cat((targets,targets_[batch_id,:,:len(list_id)]),1)
            outputs=torch.cat((outputs,outputs_[batch_id,:,:len(list_id)]),1)
            inputs=torch.cat((inputs,inputs_[batch_id,:,:len(list_id)]),1)
        targets=torch.t(targets)
        outputs=torch.t(outputs)
        inputs =torch.t(inputs)

        if mode in ['train', 'valid']:
            loss = criterion(outputs, targets)
            with torch.no_grad():
                losses.append(loss.item())

        if mode in ['train']:
            if apex:
                with amp.scale_loss(loss, optim) as scaled_loss:
                    scaled_loss.backward()
            else:
                loss.backward() # accumulate loss

            if i >= cfg.epoch_size:
                for name, param in model.named_parameters():
                    if 'bn' not in name:
                        param.grad *= 0

            if (i+1) % cfg.n_grad_acc == 0:
                optim.step() # update
                optim.zero_grad() # flush

        with torch.no_grad():
            targets_all.extend(targets.cpu().numpy())
            outputs_all.extend(torch.sigmoid(outputs).cpu().numpy())

        elapsed = int(time.time() - t1)


        # logging
        if mode in ['train']:
            if (i+1) % cfg.log_size == 0:
                eta = int(elapsed / (i+1) * (cfg.epoch_size + cfg.epoch_size_precisebn-(i+1)))
                progress = f'\r[{mode}] {i+1}/({cfg.epoch_size}+{cfg.epoch_size_precisebn}) {elapsed}(s) eta:{eta}(s) loss:{np.mean(losses):.6f} lr:{util.get_lr(optim):.2e}'
                logloss = calc_logloss(np.array(targets_all), np.array(outputs_all))

                log(progress)
                log('%.6f %s' % (logloss['logloss'], np.round(logloss['logloss_classes'], 6)))

                if writer_dict and i < cfg.epoch_size:
                    writer = writer_dict['writer']
                    global_steps = writer_dict['train_global_steps']
                    writer.add_scalar('train_loss', np.mean(losses), global_steps)
                    writer.add_scalar('LR', util.get_lr(optim), global_steps)
                    for ic in range(len(outputs_all[0])):
                        writer.add_scalar('train_logloss_{}'.format(ic), logloss['logloss_classes'][ic], global_steps)
                        for jc in range(2):
                            writer.add_scalar('train_logloss_{}_{}_mean'.format(ic,jc), logloss['logloss_matrix'][ic][jc][0], global_steps)
                    writer.add_scalar('train_logloss_sklearn', logloss['logloss'], global_steps)
                    writer_dict['train_global_steps'] = global_steps + 1

                losses = []
                ids_all = []
                targets_all = []
                outputs_all = []
        else:
            eta = int(elapsed / (i+1) * (len(loader)-(i+1)))
            progress = f'\r[{mode}] {i+1}/{len(loader)} {elapsed}(s) eta:{eta}(s) loss:{(np.sum(losses)/(i+1)):.6f} lr:{util.get_lr(optim):.2e}'
            print(progress, end='')
            sys.stdout.flush()

        if mode in ['train'] and (i+1) == (cfg.epoch_size + cfg.epoch_size_precisebn):
            with np.printoptions(precision=4, suppress=True):
                log('negative logloss:')
                log('%s' % np.round(np.array(logloss['logloss_matrix'])[:,0], 4))
                log('positive logloss:')
                log('%s' % np.round(np.array(logloss['logloss_matrix'])[:,1], 2))
            break

    result = {
        'ids': ids_all,
        'targets': np.array(targets_all),
        'outputs': np.array(outputs_all),
        'loss': np.sum(losses) / (i+1),
    }

    if mode in ['valid']:
        result.update(calc_logloss(result['targets'], result['outputs']))
        result['score'] = result['logloss']

        log(progress)
        log('%.6f %s' % (result['logloss'], np.round(result['logloss_classes'], 6)))

        with np.printoptions(precision=4, suppress=True):
            log('negative logloss:')
            log('%s' % np.round(np.array(result['logloss_matrix'])[:,0], 4))
            log('positive logloss:')
            log('%s' % np.round(np.array(result['logloss_matrix'])[:,1], 2))

        if writer_dict:
            writer = writer_dict['writer']
            global_steps = writer_dict['train_global_steps']
            writer.add_scalar('valid_loss', np.mean(losses), global_steps)
            writer.add_scalar('valid_logloss_sklearn', result['logloss'], global_steps)
            for ic in range(len(outputs_all[0])):
                writer.add_scalar('valid_logloss_{}'.format(ic), result['logloss_classes'][ic], global_steps)
                for jc in range(2):
                    writer.add_scalar('valid_logloss_{}_{}_mean'.format(ic,jc), result['logloss_matrix'][ic][jc][0], global_steps)
    else:
        log('')

    return result

def calc_logloss(targets, outputs, eps=1e-5):
    # for RSNA
    def log_loss_single(true_label, predicted, eps=1e-15):
        result = []
        for true, pred in zip(true_label, predicted):
            if true == 1:
                result.append(-ln(pred))
            else:
                result.append(-ln(1-pred))
        return np.array(result).mean(), np.array(result).sum(), len(result)


    result = []
    for i in range(6):
        index = [targets[:,i] < 0.5, targets[:, i] > 0.5]
        row = []
        for j in range(2):
            row.append(log_loss_single(np.floor(targets[index[j], i]), np.clip(outputs[index[j],i], eps, 1-eps)))
        result.append(row)

    try:
        logloss_classes = [log_loss(np.floor(targets[:,i]), np.clip(outputs[:,i], eps, 1-eps)) for i in range(6)]
    except ValueError as e:
        logloss_classes = [1, 1, 1, 1, 1, 1]

    return {
        'logloss_classes': logloss_classes,
        'logloss': np.average(logloss_classes, weights=[2,1,1,1,1,1]),
        'logloss_matrix': result
    }

def calc_auc(targets, outputs):
    try:
        macro = roc_auc_score(np.floor(targets), outputs, average='macro')
        micro = roc_auc_score(np.floor(targets), outputs, average='micro')
    except:
        macro = 1
        micro = 1
    return {
        'auc': (macro + micro) / 2,
        'auc_macro': macro,
        'auc_micro': micro,
    }



if __name__ == '__main__':

    torch.backends.cudnn.benchmark = True
    torch.backends.cudnn.deterministic = True

    try:
        main()
    except KeyboardInterrupt:
        print('Keyboard Interrupted')
