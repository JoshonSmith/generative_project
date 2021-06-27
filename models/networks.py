import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim.lr_scheduler as lr_scheduler

def get_scheduler(optimizer, cfg):
    def lambda_rule(epoch):
        lr_l = 1.0 - max(0, epoch + cfg.epoch_count - cfg.n_epochs) / float(cfg.n_epochs_decay + 1)
        return lr_l

    scheduler = lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda_rule)
    return scheduler

class ResNetBlock(nn.Module):
    def __init__(self,in_dim):
        super(ResNetBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_dim,in_dim,kernel_size=3,padding=1,bias=True)
        self.norm1 = nn.InstanceNorm2d(in_dim)
        self.conv2 = nn.Conv2d(in_dim,in_dim,kernel_size=3,padding=1,bias=True)
        self.norm2 = nn.InstanceNorm2d(in_dim)

    def forward(self,x):
        x0 = x
        x = self.conv1(x)
        x = self.norm1(x)
        x = F.relu(x,inplace=False)
        x = self.conv2(x)
        x = self.norm2(x)
        return x0 + x


class Generator(nn.Module):
    def __init__(self,input_dim,middle_dim, output_dim,num_blocks):
        super(Generator, self).__init__()
        self.layer1 = nn.Sequential(*[nn.Conv2d(input_dim, middle_dim, kernel_size=7, padding=3,bias=True),
                                  nn.InstanceNorm2d(middle_dim),nn.ReLU(True)
                                  ])

        self.dw1 = nn.Sequential(*[nn.Conv2d(middle_dim, middle_dim* 2, kernel_size=3, stride=2, padding=1,bias=True),
                                  nn.InstanceNorm2d(middle_dim*2),nn.ReLU(True)
                                  ])

        self.dw2 = nn.Sequential(*[nn.Conv2d(middle_dim*2, middle_dim * 4, kernel_size=3, stride=2, padding=1,bias=True),
                                   nn.InstanceNorm2d(middle_dim * 4), nn.ReLU(True)
                                   ])

        blocks = []
        for i in range(num_blocks):
            blocks.append(ResNetBlock(middle_dim*4))
        self.blocks = nn.Sequential(*blocks)

        self.up1 = nn.Sequential(*[nn.ConvTranspose2d(middle_dim*4, middle_dim * 2,
                                                      kernel_size=3, stride=2, padding=1,output_padding=1,bias=True),
                                   nn.InstanceNorm2d(middle_dim * 2), nn.ReLU(True)
                                   ])

        self.up2 = nn.Sequential(*[nn.ConvTranspose2d(middle_dim*2, middle_dim,
                                                      kernel_size=3, stride=2, padding=1,output_padding=1,bias=True),
                                   nn.InstanceNorm2d(middle_dim), nn.ReLU(True)
                                   ])

        self.conv2 = nn.Conv2d(middle_dim, output_dim, kernel_size=7, padding=3,bias=True)
        self.output = nn.Tanh()

    def forward(self,x,full_feat=[]):
        if len(full_feat)>0:
            features = []
            x = self.layer1(x)
            features.append(x)
            x = self.dw1(x)
            features.append(x)
            x = self.dw2(x)
            features.append(x)
            x = self.blocks(x)
            features.append(x)
            x = self.up1(x)
            features.append(x)
            x = self.up2(x)
            features.append(x)
            x = self.conv2(x)
            x = self.output(x)
            features.append(x)
            return [features[_] for _ in full_feat]
        else:
            x = self.layer1(x)
            x = self.dw1(x)
            x = self.dw2(x)
            x = self.blocks(x)
            x = self.up1(x)
            x = self.up2(x)
            x = self.conv2(x)
            x = self.output(x)
            return x


if __name__ == '__main__':
    input = torch.ones([3,3,4,4])
    model = Generator(3,2,3,2)
    output = model(input,False)
    print(output)
    output.sum().backward()


import torch
import torch.nn as nn
import torch.nn.init as init
import numpy as np
import torch.nn.functional as F

class Discriminator(nn.Module):
    def __init__(self, input_nc, dim=64, n_layers=3, norm_layer=nn.BatchNorm2d, no_antialias=False):

        super(Discriminator, self).__init__()
        self.conv1 = nn.Sequential(*[nn.Conv2d(input_nc, dim, kernel_size=4, stride=2, padding=1),
                                    nn.LeakyReLU(0.2, True)])

        kw = 4
        padw = 1
        num_mult = 1
        num_mult_prev = 1
        self.blocks = []
        for n in range(1, n_layers):  # gradually increase the number of filters
            num_mult_prev = num_mult
            num_mult = min(2 ** n, 8)
            self.blocks.append(
                nn.Sequential(*[
                    nn.Conv2d(dim * num_mult_prev, dim * num_mult, kernel_size=4, stride=2, padding=1, bias=False),
                    nn.BatchNorm2d(dim * num_mult),
                    nn.LeakyReLU(0.2, True)
                ])
            )

        self.blocks = nn.Sequential(*self.blocks)

        num_mult_prev = num_mult
        num_mult = min(2 ** n_layers, 8)
        self.last_layer = nn.Sequential(*[
            nn.Conv2d(dim* num_mult_prev, dim * num_mult, kernel_size=4, stride=1, padding=1,
                      bias=False),
            nn.BatchNorm2d(dim * num_mult),
            nn.LeakyReLU(0.2, True)
        ])
        self.classifier = nn.Sequential(
            *[nn.Conv2d(dim * num_mult, 1, kernel_size=4, stride=1, padding=1)]
        )

    def forward(self, input):
        x = self.conv1(input)
        x = self.blocks(x)
        x = self.last_layer(x)
        x = self.classifier(x)
        return x

import torch
import torch.nn as nn
import torch.nn.init as init
import numpy as np
import torch.nn.functional as F

def init_net(model:nn.Module,init_gain,gpu_ids=None):
    if not gpu_ids is None:
        model.to(gpu_ids[0])
    def init_func(m):  # define the initialization function
        classname = m.__class__.__name__
        if hasattr(m, 'weight') and m.weight is not None:
            #print(classname)
            if (classname.find('Conv') != -1 or classname.find('Linear') != -1):
                init.kaiming_normal_(m.weight.data, a=0, mode='fan_in')
            elif classname.find('Norm2d') != -1:
                init.normal_(m.weight.data, 1.0, init_gain)
                init.constant_(m.bias.data, 0.0)
    model.apply(init_func)

class PatchSampleF(nn.Module):
    def __init__(self, use_mlp=False, init_gain=0.02, nc=256, gpu_ids=[]):
        # potential issues: currently, we use the same patch_ids for multiple images in the batch
        super(PatchSampleF, self).__init__()
        self.use_mlp = use_mlp
        self.nc = nc  # hard-coded
        self.mlp_init = False
        self.init_gain = init_gain
        self.gpu_ids = gpu_ids

    def create_mlp(self, feats):
        for mlp_id, feat in enumerate(feats):
            input_dim = feat.shape[1]
            mlp = nn.Sequential(*[nn.Linear(input_dim, self.nc),
                                  nn.ReLU(), nn.Linear(self.nc, self.nc)])
            if len(self.gpu_ids) > 0:
                mlp.cuda()
            setattr(self, 'mlp_%d' % mlp_id, mlp)
        init_net(self, self.init_gain, self.gpu_ids)
        self.mlp_init = True

    def forward(self, feats, num_patches=64, patch_ids=None):
        return_ids = []
        return_feats = []
        if self.use_mlp and not self.mlp_init:
            self.create_mlp(feats)
        for feat_id, feat in enumerate(feats):
            B, H, W = feat.shape[0], feat.shape[2], feat.shape[3]
            flat_feat = feat.flatten(2, 3).permute(0, 2, 1)
            if num_patches > 0:
                if patch_ids is not None:
                    patch_id = patch_ids[feat_id]
                else:
                    patch_id = np.random.permutation(np.array([_ for _ in range(flat_feat.shape[1])]))
                    patch_id = torch.tensor(patch_id,dtype=torch.long).to(feat[0].device)
                    patch_id = patch_id[:int(min(num_patches, patch_id.shape[0]))]  # .to(patch_ids.device)
                x_sample = flat_feat[:, patch_id, :].flatten(0, 1)  # b,num_patch, c
            else:
                x_sample = flat_feat
                patch_id = []
            if self.use_mlp:
                mlp = getattr(self, 'mlp_%d' % feat_id)
                x_sample = mlp(x_sample)
            return_ids.append(patch_id)
            x_sample = F.normalize(x_sample,2.0,1)
            if num_patches == 0:
                x_sample = x_sample.permute(0, 2, 1).reshape([B, x_sample.shape[-1], H, W])
            return_feats.append(x_sample)
        return return_feats, return_ids


class PatchSampleFv2(nn.Module):
    def __init__(self, use_mlp=False, init_gain=0.02, nc=256, gpu_ids=[]):
        # potential issues: currently, we use the same patch_ids for multiple images in the batch
        super(PatchSampleFv2, self).__init__()
        self.use_mlp = use_mlp
        self.nc = nc  # hard-coded
        self.mlp_init = False
        self.init_gain = init_gain
        self.gpu_ids = gpu_ids

    def create_mlp(self, feats):
        for mlp_id, feat in enumerate(feats):
            input_dim = feat.shape[1]
            mlp = nn.Sequential(*[nn.Linear(input_dim, self.nc),
                                  nn.ReLU(), nn.Linear(self.nc, self.nc)])
            if len(self.gpu_ids) > 0:
                mlp.cuda()
            setattr(self, 'mlp_%d' % mlp_id, mlp)
        init_net(self, self.init_gain, self.gpu_ids)
        self.mlp_init = True

    def forward(self, feats, num_patches=64, patch_ids=None):
        return_ids = []
        return_feats = []
        if self.use_mlp and not self.mlp_init:
            self.create_mlp(feats)
        for feat_id, feat in enumerate(feats):
            #print('i:',feat_id,'feat_shape:',feat.shape)
            B, H, W = feat.shape[0], feat.shape[2], feat.shape[3]
            flat_feat = feat.flatten(2, 3).permute(0, 2, 1)
            if num_patches > 0:
                if patch_ids is not None:
                    patch_id = patch_ids[feat_id]
                else:
                    patch_id = np.random.permutation(np.array([_ for _ in range(flat_feat.shape[1])]))
                    patch_id = torch.tensor(patch_id,dtype=torch.long).to(feat[0].device)
                    patch_id = patch_id[:int(min(num_patches, patch_id.shape[0]))]  # .to(patch_ids.device)
                x_sample = flat_feat[:, patch_id, :]  # b,num_patch, c
            else:
                x_sample = flat_feat
                patch_id = []
            if self.use_mlp:
                mlp = getattr(self, 'mlp_%d' % feat_id)
                x_sample = mlp(x_sample)
            return_ids.append(patch_id)
            x_sample = F.normalize(x_sample,2.0,1)
            # if num_patches == 0:
            #     x_sample = x_sample.permute(0, 2, 1).reshape([B, x_sample.shape[-1], H, W])
            return_feats.append(x_sample)
            #print('x_sample:',x_sample.shape)
        return return_feats, return_ids


class GANLoss(nn.Module):
    def __init__(self, gan_mode, target_real_label=1.0, target_fake_label=0.0):
        super(GANLoss, self).__init__()
        self.register_buffer('real_label', torch.tensor(target_real_label))
        self.register_buffer('fake_label', torch.tensor(target_fake_label))
        self.loss = nn.MSELoss()
        self.gan_mode = gan_mode

    def get_target_tensor(self, prediction, target_is_real):
        if target_is_real:
            target_tensor = self.real_label
        else:
            target_tensor = self.fake_label
        return target_tensor.expand_as(prediction)

    def __call__(self, prediction, target_is_real):
        bs = prediction.size(0)
        target_tensor = self.get_target_tensor(prediction, target_is_real)
        loss = self.loss(prediction, target_tensor)
        return loss

class MappingF(nn.Module):
    def __init__(self, in_layer=4, gpu_ids=[], nc=256, patch_num=256, dim=64, init_type='normal', init_gain=0.02):
        super().__init__()
        self.init_type = init_type
        self.nc=nc
        self.dim=dim
        self.in_layer=in_layer
        self.patch_num = patch_num
        self.init_type = init_type
        self.init_gain = init_gain
        self.gpu_ids = gpu_ids
        avg = nn.AdaptiveAvgPool2d(1)
        conv = nn.Conv2d(in_layer, dim, 3, stride=2)
        self.model = nn.Sequential(*[conv, nn.ReLU(), avg, nn.Flatten(), nn.Linear(dim,dim), nn.ReLU(), nn.Linear(dim, dim)])
        init_net(self.model, self.init_gain, self.gpu_ids)

    def forward(self, x):
        x = x.view(1, -1, self.patch_num, self.nc)
        x = self.model(x)
        x_norm = F.normalize(x,2.0,1)
        return x_norm