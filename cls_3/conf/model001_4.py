workdir = './model/model001_4'
seed = 20
apex = True
traindir = './input_model001/fold4_train'
testdir  = './input_model001/fold4_test'

n_fold = 5
epoch = 30
resume_from = None

batch_size = 28
num_workers = 8
imgsize = (60,)
loss = dict(
    name='BCEWithLogitsLoss',
    params=dict(),
)

optim = dict(
    name='Adam',
    params=dict(
        lr=6e-5,
    ),
)

scheduler1 = dict(
    name='MultiStepLR',
    params=dict(
        milestones=[5,10],
        gamma=2/3,
    ),
)
scheduler2 = dict(
    name='CosineAnnealingLR',
    params=dict(
        T_max=epoch
    ),
)

scheduler = scheduler2

#normalize = {'mean': [0.485, 0.456, 0.406], 'std': [0.229, 0.224, 0.225],}
normalize = None

totensor = dict(name='ToTensor', params=dict(normalize=normalize))

data = dict(
    train=dict(
        dataset_type='CustomDataset',
        annotations='./cache/train_folds_s50.pkl',
        imgdir=traindir,
        imgsize=imgsize,
        n_grad_acc=1,
        loader=dict(
            shuffle=True,
            batch_size=batch_size,
            drop_last=True,
            num_workers=num_workers,
            pin_memory=True,
        ),
        dataset_policy='all',
        epoch_size=600,
        epoch_size_precisebn=0,
        transforms=[totensor],
        log_size=100
    ),
    valid = dict(
        dataset_type='CustomDataset',
        annotations='./cache/train_folds_s50.pkl',
        imgdir=traindir,
        imgsize=imgsize,
        loader=dict(
            shuffle=False,
            batch_size=batch_size,
            drop_last=False,
            num_workers=num_workers,
            pin_memory=False,
        ),
        transforms=[totensor],
        dataset_policy='all',
    ),
    test = dict(
        dataset_type='CustomDataset',
        annotations='./cache/test.pkl',
        imgdir=testdir,
        imgsize=imgsize,
        loader=dict(
            shuffle=False,
            batch_size=batch_size,
            drop_last=False,
            num_workers=num_workers,
            pin_memory=False,
        ),
        transforms=[totensor],
        dataset_policy='all',
    ),
)
