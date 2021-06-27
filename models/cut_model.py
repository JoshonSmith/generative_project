import numpy as np
import torch
from .base_model import BaseModel
from . import networks
from .patchnce import PatchNCELoss
import util.util as util

import os
from os.path import join


class CUTModel(BaseModel):
    def __init__(self, cfg):
        super(CUTModel, self).__init__(cfg)
        self.loss_names = ['G_GAN', 'D_real', 'D_fake', 'G', 'NCE']
        self.visual_names = ['real_A', 'fake_B', 'real_B']
        self.nce_layers = [int(i) for i in self.cfg.nce_layers.split(',')]

        if cfg.nce_idt and self.isTrain:
            self.loss_names += ['NCE_Y']
            self.visual_names += ['idt_B']

        if self.isTrain:
            self.model_names = ['G', 'F', 'D']
        else:  # during test time, only load G
            self.model_names = ['G']

        self.netG = networks.Generator(cfg.input_nc,64,cfg.output_nc,9)
        networks.init_net(self.netG,init_gain=0.02,gpu_ids=self.gpu_ids)
        self.netF = networks.PatchSampleF(use_mlp=True,nc=cfg.netF_nc,gpu_ids=self.gpu_ids)

        self.optimizers = []
        self.image_paths = []
        self.metric = 0 
        if self.isTrain:
            self.netD = networks.Discriminator(
                input_nc=cfg.output_nc,dim=cfg.ndf,n_layers=cfg.n_layers_D
            )
            networks.init_net(self.netD,init_gain=0.02,gpu_ids=self.gpu_ids)
            # define loss functions
            self.criterionGAN = networks.GANLoss(cfg.gan_mode).to(self.device)
            self.criterionNCE = []

            for nce_layer in self.nce_layers:
                self.criterionNCE.append(PatchNCELoss(cfg).to(self.device))

            self.criterionIdt = torch.nn.L1Loss().to(self.device)
            self.optimizer_G = torch.optim.Adam(self.netG.parameters(), lr=cfg.lr, betas=(cfg.beta1, cfg.beta2))
            self.optimizer_D = torch.optim.Adam(self.netD.parameters(), lr=cfg.lr, betas=(cfg.beta1, cfg.beta2))
            self.optimizers.append(self.optimizer_G)
            self.optimizers.append(self.optimizer_D)

    def data_dependent_initialize(self, data):
        self.set_input(data)
        bs_per_gpu = self.real_A.size(0) // max(len(self.cfg.gpu_ids), 1)
        self.real_A = self.real_A[:bs_per_gpu]
        self.real_B = self.real_B[:bs_per_gpu]
        self.forward()                     # compute fake images: G(A)
        if self.cfg.isTrain:
            self.compute_D_loss().backward()                  # calculate gradients for D
            self.compute_G_loss().backward()                   # calculate graidents for G
            if self.cfg.lambda_NCE > 0.0:
                self.optimizer_F = torch.optim.Adam(self.netF.parameters(), lr=self.cfg.lr, betas=(self.cfg.beta1, self.cfg.beta2))
                self.optimizers.append(self.optimizer_F)

    def optimize_parameters(self):
        self.forward()
        self.set_requires_grad(self.netD, True)
        self.optimizer_D.zero_grad()
        self.loss_D = self.compute_D_loss()
        self.loss_D.backward()
        self.optimizer_D.step()

        self.set_requires_grad(self.netD, False)
        self.optimizer_G.zero_grad()
        self.optimizer_F.zero_grad()
        self.loss_G = self.compute_G_loss()
        self.loss_G.backward()
        self.optimizer_G.step()
        self.optimizer_F.step()

    def set_input(self, input):
        AtoB = self.cfg.direction == 'AtoB'
        self.real_A = input['A' if AtoB else 'B'].to(self.device)
        self.real_B = input['B' if AtoB else 'A'].to(self.device)
        self.image_paths = input['A_paths' if AtoB else 'B_paths']

    def forward(self):
        self.real = torch.cat((self.real_A, self.real_B), dim=0) if self.cfg.nce_idt and self.cfg.isTrain else self.real_A
        if self.cfg.flip_equivariance:
            self.flipped_for_equivariance = self.cfg.isTrain and (np.random.random() < 0.5)
            if self.flipped_for_equivariance:
                self.real = torch.flip(self.real, [3])

        self.fake = self.netG(self.real)
        self.fake_B = self.fake[:self.real_A.size(0)]
        if self.cfg.nce_idt:
            self.idt_B = self.fake[self.real_A.size(0):]

    def compute_D_loss(self):
        fake = self.fake_B.detach()
        pred_fake = self.netD(fake)
        self.loss_D_fake = self.criterionGAN(pred_fake, False).mean()
        self.pred_real = self.netD(self.real_B)
        loss_D_real = self.criterionGAN(self.pred_real, True)
        self.loss_D_real = loss_D_real.mean()
        self.loss_D = (self.loss_D_fake + self.loss_D_real) * 0.5
        return self.loss_D

    def compute_G_loss(self):
        fake = self.fake_B
        if self.cfg.lambda_GAN > 0.0:
            pred_fake = self.netD(fake)
            self.loss_G_GAN = self.criterionGAN(pred_fake, True).mean() * self.cfg.lambda_GAN
        else:
            self.loss_G_GAN = 0.0

        if self.cfg.lambda_NCE > 0.0:
            self.loss_NCE = self.calculate_NCE_loss(self.real_A, self.fake_B)
        else:
            self.loss_NCE, self.loss_NCE_bd = 0.0, 0.0

        if self.cfg.nce_idt and self.cfg.lambda_NCE > 0.0:
            self.loss_NCE_Y = self.calculate_NCE_loss(self.real_B, self.idt_B)
            loss_NCE_both = (self.loss_NCE + self.loss_NCE_Y) * 0.5
        else:
            loss_NCE_both = self.loss_NCE

        self.loss_G = self.loss_G_GAN + loss_NCE_both
        return self.loss_G

    def calculate_NCE_loss(self, src, tgt):
        n_layers = len(self.nce_layers)
        feat_q = self.netG(tgt, self.nce_layers)

        if self.cfg.flip_equivariance and self.flipped_for_equivariance:
            feat_q = [torch.flip(fq, [3]) for fq in feat_q]

        feat_k = self.netG(src, self.nce_layers)
        feat_k_pool, sample_ids = self.netF(feat_k, self.cfg.num_patches, None)
        feat_q_pool, _ = self.netF(feat_q, self.cfg.num_patches, sample_ids)

        total_nce_loss = 0.0
        for f_q, f_k, crit, nce_layer in zip(feat_q_pool, feat_k_pool, self.criterionNCE, self.nce_layers):
            loss = crit(f_q, f_k,self.cfg.weighted) * self.cfg.lambda_NCE
            total_nce_loss += loss.mean()

        return total_nce_loss / n_layers
