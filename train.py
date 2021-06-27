import sys 
import argparse
import os
from os.path import join, isdir, isfile, exists, dirname, abspath
import time 

import torch
import torch.nn as nn
import torch.nn.functional as F
from tensorboardX import SummaryWriter

import yaml 
from easydict import EasyDict 

import data
import models
from util.util import Logger, tensor2im, mkdirs, Timer

def update_config(args):
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
    print('--------- model {} specific params ---------'.format(args.model))
    print(model_cfg)
    print('---------------------------------------------')
    for k, v in model_cfg.items():
        cfg[k] = v
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

def create_logger(cfg):
    experiment_dir = join(cfg.checkpoints_dir, cfg.name)
    logger = Logger(experiment_dir)
    return logger

def main(args):
    cfg = update_config(args)
    dataloader, dataset_size = create_dataset(cfg)
    model = create_model(cfg)
    logger = create_logger(cfg)
    cfg.logger = logger
    timer = Timer(['data', 'optimize', 'epoch'])

    iter_total = 0
    times = []
    start_epoch, end_epoch = cfg.start_epoch, cfg.n_epochs + cfg.n_epochs_decay
    for e in range(start_epoch, end_epoch + 1):
        timer.tik('epoch')
        dataloader.dataset.update_epoch(e)
        iter_epoch = 0
        for i, data in enumerate(dataloader):
            timer.tik('data')
            bsize = data['A'].size(0)
            iter_total += bsize
            iter_epoch += bsize
            if len(cfg.gpu_ids):
                torch.cuda.synchronize()
            timer.tik('optimize')
            if e == cfg.start_epoch and i == 0:
                model.data_dependent_initialize(data)
                model.setup(cfg)
                model.parallelize()
            model.set_input(data)
            model.optimize_parameters()
            if len(cfg.gpu_ids):
                torch.cuda.synchronize()
            timer.tok('optimize')

            if iter_total % cfg.display_freq == 0:
                model.compute_visuals()
                for name, img_tensor in model.get_current_visuals().items():
                    img_npy = tensor2im(img_tensor.detach()).transpose(2, 0, 1)
                    logger.add_image(name, img_npy, iter_total)
            
            if iter_total % cfg.print_freq == 0:    
                losses = model.get_current_losses()
                logger.add_loss(e, iter_epoch, losses, timer['optimize'], timer['data'])
                logger.add_scalars('losses', losses, iter_total)

            timer.tok('data')

        if e % cfg.save_epoch_freq == 0:
            print('save mdoel @ epoch {} end, iters {}'.format(e, iter_total))
            model.save_networks('latest')
            model.save_networks(e)
            
        timer.tok('epoch')
        print('Epoch: ({:d}/{:d}) time {:.3f}'.format(e, end_epoch, timer['epoch']))
        model.update_learning_rate()             



if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='argument parse for train scripts')
    parser.add_argument('--dataroot', type=str)
    parser.add_argument('--gpu_ids', default='0', type=str)
    parser.add_argument('--name', default='', type=str)
    parser.add_argument('--model', default='cut', type=str)
    parser.add_argument('--phase', default='train', type=str, help='train (or val/test) set')
    parser.add_argument('--isTrain', action='store_false', default=True, help='train (or test)')
    parser.add_argument('--num_threads', default=4, type=int)
    # resume args
    #  parser.add_argument('--save_by_iter', action='store_true')
    parser.add_argument('--start_epoch', default=0, type=int, help='not 0 if resume model')
    parser.add_argument('--continue_train', action='store_true')
    parser.add_argument('--pretrained_name', type=str, default=None, help='resume model names')
    parser.add_argument('--epoch', type=str, default='latest', help='resume model epoch')
    parser.add_argument('--verbose', action='store_true', help='True with more debug info')
    parser.add_argument('--weighted',type=bool,default=False,help='weight cut')
    parser.add_argument('--prob_weight', type=bool, default=False,help='prob weight cut, weight cut must be True')
    args = parser.parse_args()
    
    main(args)