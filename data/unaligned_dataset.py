import os
from os.path import join, exists, isdir
from PIL import Image
import random
import util.util as util
import torchvision.transforms as transforms
import torch.utils.data as data

def make_dataset(img_dir):
    img_path_list = []
    assert isdir(img_dir), '{} must be a valid directory'.format(img_dir)
    for root, _, fnames in sorted(os.walk(img_dir)):
        for fname in fnames:
            if fname.endswith(('jpg', 'png')):
                img_path = join(root, fname)
                img_path_list.append(img_path)
    return img_path_list 

def get_transform(cfg, finetune):
    transform_list = []
    if 'resize' in cfg.preprocess:
        # in finetune stage: resize to crop size (random crop not works)
        target_size = cfg.crop_size if finetune else cfg.load_size
        transform_list.append(transforms.Resize((target_size, target_size), Image.BICUBIC))

    if 'crop' in cfg.preprocess:
        transform_list.append(transforms.RandomCrop(cfg.crop_size))

    transform_list.extend([
        transforms.Lambda(lambda img: __make_power_2(img, base=4)),
        transforms.ToTensor(), 
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
    ])
    return transforms.Compose(transform_list)

def __make_power_2(img, base):
    nsize = [int(round(os / base) * base) for os in img.size]
    nw, nh = nsize
    ow, oh = img.size

    if nh == oh and nw == ow:
        return img
    else:
        return img.resize((nw, nh), Image.BICUBIC)


class UnalignedDataset(data.Dataset):
    def __init__(self, cfg):
        super(UnalignedDataset, self).__init__()
        self.cfg = cfg
        self.root = cfg.dataroot
        self.current_epoch = 0
        self.phase = cfg.phase
        for suffix in ['A', 'B']:
            dir_path = join(self.root, self.phase + suffix)
            if not exists(dir_path) and self.phase == 'test':
                dir_path = join(self.root, 'val' + suffix)
            setattr(self, 'dir_' + suffix, dir_path)
            img_path_list = sorted(make_dataset(dir_path))
            setattr(self, '{}_paths'.format(suffix), img_path_list)
            setattr(self, '{}_size'.format(suffix), len(img_path_list))

    def __getitem__(self, index):
        '''
        index: int, random integer for data indexing
        output: A, B (tensor); A_paths, B_paths (str)
        '''
        out_dict = {}
        transform = get_transform(self.cfg, self.cfg.isTrain and self.current_epoch > self.cfg.n_epochs)
        for suffix in ['A', 'B']:
            total_len = getattr(self, '{}_size'.format(suffix))
            idx = index % total_len
            if suffix == 'B' and not self.cfg.serial_batches:
                idx = random.randint(0, total_len - 1)
            img_path = getattr(self, '{}_paths'.format(suffix))[idx]
            img = Image.open(img_path).convert('RGB')
            img = transform(img)
            out_dict[suffix] = img
            out_dict['{}_paths'.format(suffix)] = img_path
        return out_dict

    def update_epoch(self, cur_epoch):
        self.current_epoch = cur_epoch
            
    def __len__(self):
        return max(self.A_size, self.B_size)
        