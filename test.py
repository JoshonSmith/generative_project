import sys
import argparse
import os
from os.path import join, dirname, isdir, isfile, exists, abspath

import torch
import torch.nn as nn
import torch.nn.functional as F

import yaml 
from easydict import EasyDict
import data
import models
# import util.util as util
import util.html as html

def reset_config(args):
    this_dir = dirname(abspath(__file__))
    default_cfg_path = join(this_dir, 'configs/default_configs.yaml')
    cfg = yaml.load(open(default_cfg_path))
    cfg = EasyDict(cfg)
    # add attr from args
    for k, v in args._get_kwargs():
        if k == 'gpu_ids': v = [int(vid) for vid in v.split(',')]
        cfg[k] = v 
    # add attr from model specific 
    model_cfg_path = join(this_dir, 'configs/{}.yaml'.format(args.model))
    model_cfg = yaml.load(open(model_cfg_path))
    model_cfg = EasyDict(model_cfg)
    for k, v in model_cfg.items():
        cfg[k] = v

    # reset attr fixed for test
    cfg.num_threads = 0
    cfg.batch_size = 1
    cfg.serial_batches = True
    cfg.no_flip = True
    cfg.isTrain = False # test fixed
    return cfg

def create_dataset(cfg):
    dataset = eval('data.' + cfg.dataset_mode.capitalize() + 'Dataset')(cfg)
    print('dataset [{}] was created'.format(type(dataset).__name__))
    dataset_size = len(dataset)
    print('The number of training images = {}'.format(dataset_size))

    dataloader = torch.utils.data.DataLoader(
        dataset, batch_size=cfg.batch_size, 
        shuffle=True if cfg.isTrain else False,
        num_workers=int(cfg.num_threads), 
        drop_last=True if cfg.phase=='train' else False,
    )
    return dataloader, dataset_size

def create_model(cfg):
    model = eval('models.' + cfg.model.upper() + 'Model')(cfg)
    return model

def main(args):
    cfg = reset_config(args)
    dataset, dataset_size = create_dataset(cfg)
    model = create_model(cfg)

    vis_dir = join(cfg.results_dir, cfg.name, '{}_{}'.format(cfg.phase, cfg.epoch)) 
    print('creating web directory {}'.format(vis_dir))
    vispage = html.HTML(vis_dir, 'Experiment = {}, Phase = {}, Epoch = {}'.format(cfg.name, cfg.phase, cfg.epoch))

    for i, data in enumerate(dataset):
        if i == 0:
            model.data_dependent_initialize(data)
            model.setup(cfg)               
            model.parallelize()
            model.eval()
       
        model.set_input(data)  
        model.test()          
        visuals = model.get_current_visuals() 
        img_path = model.get_image_paths() 
        if i % 5 == 0:
            print('processing (%04d)-th image... %s' % (i, img_path))
        vispage.save_images(visuals, img_path, width=cfg.display_winsize)
    vispage.save() 


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='argument parse for test scripts')
    parser.add_argument('--dataroot', type=str)
    parser.add_argument('--gpu_ids', default='0', type=str)
    parser.add_argument('--name', default='', type=str)
    parser.add_argument('--model', default='cut', type=str)
    parser.add_argument('--phase', default='train', type=str, help='train (or val/test) set')
    parser.add_argument('--results_dir', default='./results', type=str)
    parser.add_argument('--epoch', type=str, default='latest', help='resume model epoch')
    parser.add_argument('--verbose', action='store_true', help='True with more debug info')

    args = parser.parse_args()
    main(args)