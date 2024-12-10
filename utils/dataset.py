"""
Modified from https://github.com/ssfgunner/SNELL
"""

import os
import torch

from torchvision import transforms
from torchvision.datasets.folder import ImageFolder, default_loader

from timm.data.transforms import str_to_interp_mode, RandomResizedCropAndInterpolation
from timm.data.constants import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD
from timm.data import create_transform

class general_dataset(ImageFolder):
    def __init__(self, root, train=True, transform=None, test=False):
        self.dataset_root = root
        self.loader = default_loader
        self.target_transform = None
        self.transform = transform

        if not test:
            train_list_path = os.path.join(self.dataset_root, 'train800.txt')
            test_list_path = os.path.join(self.dataset_root, 'val200.txt')
        else:
            train_list_path = os.path.join(self.dataset_root, 'train800val200.txt')
            test_list_path = os.path.join(self.dataset_root, 'test.txt')

        self.samples = []
        if train:
            with open(train_list_path, 'r') as f:
                for line in f:
                    img_name = line.split(' ')[0]
                    label = int(line.split(' ')[1])
                    self.samples.append((os.path.join(root,img_name), label))
        else:
            with open(test_list_path, 'r') as f:
                for line in f:
                    img_name = line.split(' ')[0]
                    label = int(line.split(' ')[1])
                    self.samples.append((os.path.join(root,img_name), label))


def build_dataset(is_train, args, multiview=False):
    transform = build_transform(is_train, args, multiview)
   
    if args.data_set == 'clevr_count':
        dataset = general_dataset(args.data_path, train=is_train, transform=transform, test=args.test)
        nb_classes = 8
    elif args.data_set == 'diabetic_retinopathy':
        dataset = general_dataset(args.data_path, train=is_train, transform=transform, test=args.test)
        nb_classes = 5
    elif args.data_set == 'dsprites_loc':
        dataset = general_dataset(args.data_path, train=is_train, transform=transform, test=args.test)
        nb_classes = 16
    elif args.data_set == 'dtd':
        dataset = general_dataset(args.data_path, train=is_train, transform=transform, test=args.test)
        nb_classes = 47
    elif args.data_set == 'kitti':
        dataset = general_dataset(args.data_path, train=is_train, transform=transform, test=args.test)
        nb_classes = 4
    elif args.data_set == 'oxford_iiit_pet':
        dataset = general_dataset(args.data_path, train=is_train, transform=transform, test=args.test)
        nb_classes = 37
    elif args.data_set == 'resisc45':
        dataset = general_dataset(args.data_path, train=is_train, transform=transform, test=args.test)
        nb_classes = 45
    elif args.data_set == 'smallnorb_ele':
        dataset = general_dataset(args.data_path, train=is_train, transform=transform, test=args.test)
        nb_classes = 9
    elif args.data_set == 'svhn':
        dataset = general_dataset(args.data_path, train=is_train, transform=transform, test=args.test)
        nb_classes = 10
    elif args.data_set == 'cifar':
        dataset = general_dataset(args.data_path, train=is_train, transform=transform, test=args.test)
        nb_classes = 100
    elif args.data_set == 'clevr_dist':
        dataset = general_dataset(args.data_path, train=is_train, transform=transform, test=args.test)
        nb_classes = 6
    elif args.data_set == 'caltech101':
        dataset = general_dataset(args.data_path, train=is_train, transform=transform, test=args.test)
        nb_classes = 102
    elif args.data_set == 'dmlab':
        dataset = general_dataset(args.data_path, train=is_train, transform=transform, test=args.test)
        nb_classes = 6
    elif args.data_set == 'dsprites_ori':
        dataset = general_dataset(args.data_path, train=is_train, transform=transform, test=args.test)
        nb_classes = 16
    elif args.data_set == 'eurosat':
        dataset = general_dataset(args.data_path, train=is_train, transform=transform, test=args.test)
        nb_classes = 10
    elif args.data_set == 'oxford_flowers102':
        dataset = general_dataset(args.data_path, train=is_train, transform=transform, test=args.test)
        nb_classes = 102
    elif args.data_set == 'patch_camelyon':
        dataset = general_dataset(args.data_path, train=is_train, transform=transform, test=args.test)
        nb_classes = 2
    elif args.data_set == 'smallnorb_azi':
        dataset = general_dataset(args.data_path, train=is_train, transform=transform, test=args.test)
        nb_classes = 18
    elif args.data_set == 'sun397':
        dataset = general_dataset(args.data_path, train=is_train, transform=transform, test=args.test)
        nb_classes = 397

    return dataset, nb_classes

# Based on GPS https://github.com/FightingFighting/GPS/blob/main/data/transforms_factory.py
class TwoCropTransform:
    """Create two crops of the same iamges"""
    def __init__(self, transform):
        self.transform = transform
    
    def __call__(self, x):
        return [self.transform(x), self.transform(x)]
    
def transforms_contrastaug_train(
    img_size=224,
    scale=None,
    ratio=None,
    hflip=0.5,
    color_jitter=0.8,
    gray_scale=0.2,
    interpolation='bilinear',
    mean=[0.5,0.5,0.5],
    std=[0.5,0.5,0.5]
):
    tfl = [
        RandomResizedCropAndInterpolation(img_size, scale=scale, ratio=ratio, interpolation=interpolation),
        transforms.RandomHorizontalFlip(p=hflip)
    ]
    tfl += [
        transforms.RandomApply([
            transforms.ColorJitter(0.4, 0.4, 0.4, 0.1)
        ], p=color_jitter),
        transforms.RandomGrayscale(p=gray_scale),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=torch.tensor(mean),
            std=torch.tensor(std))
    ]
    return TwoCropTransform(transforms.Compose(tfl))
    
def build_transform(is_train, args, multiview=False):
    
    # multi-view input for contrastive loss 
    if multiview:
        return transforms_contrastaug_train(
            img_size=args.input_size,
            scale=[0.08, 1.0], ratio=[3./4., 4./3.],
            interpolation='bicubic'
        )

    if not args.no_aug and is_train:
        transform = create_transform(
            input_size=args.input_size,
            is_training=True,
            interpolation=args.train_interpolation,
            hflip=0.5,
            scale=[0.08, 1.0], ratio=[3./4., 4./3.]
        )
        return transform
    else: 
        return transforms.Compose([
            transforms.Resize((args.input_size, args.input_size), interpolation=str_to_interp_mode(args.train_interpolation)),
            transforms.ToTensor(),
            transforms.Normalize(IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD)
        ])