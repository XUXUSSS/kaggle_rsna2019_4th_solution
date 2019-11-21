workdir = './model/model001'
seed = 20
apex = True

n_fold = 5
epoch = 30
resume_from = None

batch_size = 28
num_workers = 8
imgsize = (448, 448) #(height, width)

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

model1 = dict(
    name='resnet50',
    pretrained='imagenet',
    n_output=6,
    dropout=None,
    initialbn=True
)

model2 = dict(
    name='resnext101_32x8d_swsl',
    pretrained='SWSL',
    n_output=6,
    initialbn=True
)

model3 = dict(
    name='se_resnext50_32x4d',
    pretrained='imagenet',
    n_output=6,
    initialbn=True
)

model4 = dict(
    name='efficientnetB4',
    pretrained='imagenet',
    n_output=6,
    initialbn=True
)

model5 = dict(
    name='resnext50_32x4d_swsl',
    pretrained='imagenet',
    n_output=6,
    initialbn=True
)

model = model5

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

crop = dict(name='RandomResizedCrop', params=dict(height=imgsize[0], width=imgsize[1], scale=(0.7,1.0), p=1.0))
resize = dict(name='Resize', params=dict(height=imgsize[0], width=imgsize[1]))
hflip = dict(name='HorizontalFlip', params=dict(p=0.5,))
vflip = dict(name='VerticalFlip', params=dict(p=0.5,))
contrast = dict(name='RandomBrightnessContrast', params=dict(brightness_limit=0.08, contrast_limit=0.08, p=0.5))
totensor = dict(name='ToTensor', params=dict(normalize=normalize))
rotate = dict(name='Rotate', params=dict(limit=30, border_mode=0), p=0.7)

window_policy = dict(
    index = 2,
    noise = True,
    range=[[4,7,4],[5,10,15]],
    isuniform=True,
    drop=False
)

mytransforms_train = dict(
    apply = True,
    pixeljitter = dict(apply=True, range=0.05, isuniform=True),
    channeljitter = dict(apply=False, range=[0.05, 0.05, 0.05], isuniform=True)
)

data = dict(
    train=dict(
        dataset_type='CustomDataset',
        annotations='./cache/train_folds_s10.pkl',
        imgdir='./input/stage_1_train_images',
        imgsize=imgsize,
        n_grad_acc=1,
        loader=dict(
            shuffle=True,
            batch_size=batch_size,
            drop_last=True,
            num_workers=num_workers,
            pin_memory=True,
        ),
        transforms=[crop, hflip, rotate, contrast, totensor],
        dataset_policy='all',
        window_policy= window_policy,
        mytransforms=mytransforms_train,
        epoch_size=3600,
        epoch_size_precisebn=500,
        log_size=100
    ),
    valid = dict(
        dataset_type='CustomDataset',
        annotations='./cache/train_folds_s10.pkl',
        imgdir='./input/stage_1_train_images',
        imgsize=imgsize,
        loader=dict(
            shuffle=False,
            batch_size=batch_size,
            drop_last=False,
            num_workers=num_workers,
            pin_memory=False,
        ),
        transforms=[crop, hflip, rotate, contrast, totensor],
        dataset_policy='all',
        window_policy=dict(index=2, noise=False, drop=False),
        mytransforms=dict(apply=False)
    ),
    test = dict(
        dataset_type='CustomDataset',
        annotations='./cache/test.pkl',
        imgdir='./input/stage_1_test_images',
        imgsize=imgsize,
        loader=dict(
            shuffle=False,
            batch_size=batch_size,
            drop_last=False,
            num_workers=num_workers,
            pin_memory=False,
        ),
        transforms=[crop, hflip, rotate, contrast, totensor],
        dataset_policy='all',
        window_policy=dict(index=2, noise=False, drop=False),
        mytransforms=dict(apply=False)
    ),
)
