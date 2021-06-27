import os
from os.path import join, dirname, exists
import time
from argparse import Namespace
from tensorboardX import SummaryWriter
import numpy as np
from PIL import Image

def tensor2im(img_tensor):
    if len(img_tensor.size()) == 4:
        img_tensor = img_tensor[0] # vis first img in batch
    img_npy = img_tensor.clamp(-1, 1).cpu().float().numpy()
    if img_npy.shape[0] == 1:
        img_npy = np.tile(img_npy, (3, 1, 1))
    img_npy = np.transpose(img_npy,  (1, 2, 0))
    img_npy = (img_npy + 1) / 2.0 * 255
    return img_npy.astype(np.uint8)

def mkdirs(paths):
    if isinstance(paths, str):
        os.makedirs(paths)
    elif isinstance(paths, list):
        for path in paths:
            os.makedirs(path)

def modify_attr(cfg, **kwargs):
    conf = Namespace(**vars(cfg))
    for k in kwargs:
        setattr(cfg, k, kwargs[k])
    return conf

def save_image(img_npy, img_path, aspect_ratio=1.0):
    h, w, _ = img_npy.shape
    img_pil = Image.fromarray(img_npy)

    if aspect_ratio is not None:
        nsize = (h, int(w * aspect_ratio)) if aspect_ratio > 1.0 \
            else (int(h / aspect_ratio), w)
        img_pil = img_pil.resize(nsize, Image.BICUBIC)

    img_pil.save(img_path)

class Logger(SummaryWriter):
    def __init__(self, log_dir):
        super().__init__(log_dir)
        if not exists(log_dir):
            os.makedirs(log_dir)
        
        self.log_path = join(log_dir, 'loss_log.txt')
        with open(self.log_path, 'a') as f:
            now = time.strftime('%c')
            f.write('================ Training Loss ({}) ================\n'.format(now))
        f.close()

    # losses: same format as |losses| of plot_current_losses
    def add_loss(self, epoch, iters, losses, t_comp, t_data):
        info = '(epoch: %d, iters: %d, time: %.3f, data: %.3f) ' % (epoch, iters, t_comp, t_data)
        for k, v in losses.items():
            info += '%s: %.3f ' % (k, v)

        print(info)  
        with open(self.log_path, "a") as f:
            f.write('{}\n'.format(info))  
        f.close()

class Timer(object):
    def __init__(self, names):
        self.names = names
        self.times = {}
        self.reset()
    
    def reset(self):
        # tuple: (last_tik, last_tok - last_tik)
        for n in self.names:
            self.times[n] = {'tik': 0, 'duration': 0}
    
    def tik(self, name):
        self.times[name]['tik'] = time.time()

    def tok(self, name):
        self.times[name]['duration'] = time.time() - self.times[name]['tik']

    def __getitem__(self, name):
        return self.times[name]['duration']

