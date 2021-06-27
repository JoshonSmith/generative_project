import itertools
import torch
from .base_model import BaseModel
from . import networks
from .patchnce import PatchNCELoss_bicut as PatchNCELoss

import random
import torch


class ImagePool():
    def __init__(self, pool_size):
        self.pool_size = pool_size
        if self.pool_size > 0:  # create an empty pool
            self.num_imgs = 0
            self.images = []

    def query(self, images):
        if self.pool_size == 0:  # if the buffer size is 0, do nothing
            return images
        return_images = []
        for image in images:
            image = torch.unsqueeze(image.data, 0)
            if self.num_imgs < self.pool_size:   # if the buffer is not full; keep inserting current images to the buffer
                self.num_imgs = self.num_imgs + 1
                self.images.append(image)
                return_images.append(image)
            else:
                p = random.uniform(0, 1)
                if p > 0.5:  # by 50% chance, the buffer will return a previously stored image, and insert the current image into the buffer
                    random_id = random.randint(0, self.pool_size - 1)  # randint is inclusive
                    tmp = self.images[random_id].clone()
                    self.images[random_id] = image
                    return_images.append(tmp)
                else:       # by another 50% chance, the buffer will return the current image
                    return_images.append(image)
        return_images = torch.cat(return_images, 0)   # collect all the images and return
        return return_images


class BiCUTModel(BaseModel):
    def __init__(self, cfg):
        BaseModel.__init__(self, cfg)

        # specify the training losses you want to print out.
        # The training/test scripts will call <BaseModel.get_current_losses>
        self.loss_names = ['D_A', 'G_A', 'NCE1', 'D_B', 'G_B', 'NCE2', 'G', 'Sim']
        visual_names_A = ['real_A', 'fake_B']
        visual_names_B = ['real_B', 'fake_A']
        self.nce_layers = [int(i) for i in self.cfg.nce_layers.split(',')]

        if cfg.nce_idt and self.isTrain:
            self.loss_names += ['idt_B', 'idt_A']
            visual_names_A.append('idt_A')
            visual_names_B.append('idt_B')

        self.visual_names = visual_names_A + visual_names_B  # combine visualizations for A and B

        if self.isTrain:
            self.model_names = ['G_A', 'F1', 'D_A', 'G_B', 'F2', 'D_B', 'F3', 'F4', 'F5', 'F6']
        else:  # during test time, only load G
            self.model_names = ['G_A', 'G_B']

        # define networks (both generator and discriminator)
        self.netG_A = networks.Generator(cfg.input_nc, 64, cfg.output_nc, 9)
        networks.init_net(self.netG_A, init_gain=0.02, gpu_ids=self.gpu_ids)
        self.netG_B = networks.Generator(cfg.input_nc, 64, cfg.output_nc, 9)
        networks.init_net(self.netG_B, init_gain=0.02, gpu_ids=self.gpu_ids)
        self.netF1 = networks.PatchSampleF(use_mlp=True,nc=cfg.netF_nc,gpu_ids=self.gpu_ids)
        self.netF2 = networks.PatchSampleF(use_mlp=True,nc=cfg.netF_nc,gpu_ids=self.gpu_ids)
        n_layers = len(self.nce_layers)
        self.netF3 =  networks.MappingF(n_layers, gpu_ids=self.gpu_ids)
        self.netF4 = networks.MappingF(n_layers, gpu_ids=self.gpu_ids)
        self.netF5 = networks.MappingF(n_layers, gpu_ids=self.gpu_ids)
        self.netF6 = networks.MappingF(n_layers, gpu_ids=self.gpu_ids)

        if self.isTrain:
            self.netD_A = networks.Discriminator(
                input_nc=cfg.output_nc, dim=cfg.ndf, n_layers=cfg.n_layers_D
            )
            networks.init_net(self.netD_A, init_gain=0.02, gpu_ids=self.gpu_ids)
            self.netD_B = networks.Discriminator(
                input_nc=cfg.output_nc, dim=cfg.ndf, n_layers=cfg.n_layers_D
            )
            networks.init_net(self.netD_B, init_gain=0.02, gpu_ids=self.gpu_ids)
            self.fake_A_pool = ImagePool(cfg.pool_size)  # create image buffer to store previously generated images
            self.fake_B_pool = ImagePool(cfg.pool_size)  # create image buffer to store previously generated images
            # define loss functions
            self.criterionGAN = networks.GANLoss(cfg.gan_mode).to(self.device)
            self.criterionNCE = []

            for nce_layer in self.nce_layers:
                self.criterionNCE.append(PatchNCELoss(cfg).to(self.device))

            self.criterionIdt = torch.nn.L1Loss().to(self.device)
            self.criterionSim = torch.nn.L1Loss('sum').to(self.device)
            self.optimizer_G = torch.optim.Adam(itertools.chain(self.netG_A.parameters(), self.netG_B.parameters()),
                                                lr=self.cfg.lr, betas=(cfg.beta1, cfg.beta2))
            self.optimizer_D = torch.optim.Adam(itertools.chain(self.netD_A.parameters(), self.netD_B.parameters()),
                                                lr=self.cfg.lr, betas=(cfg.beta1, cfg.beta2))
            self.optimizers.append(self.optimizer_G)
            self.optimizers.append(self.optimizer_D)

    def data_dependent_initialize(self, data):
        self.set_input(data)
        bs_per_gpu = self.real_A.size(0) // max(len(self.cfg.gpu_ids), 1)
        self.real_A = self.real_A[:bs_per_gpu]
        self.real_B = self.real_B[:bs_per_gpu]
        self.forward()  # compute fake images: G(A)
        if self.cfg.isTrain:
            self.compute_G_loss().backward()  # calculate graidents for G
            self.backward_D_A()  # calculate gradients for D_A
            self.backward_D_B()  # calculate graidents for D_B
            if self.cfg.lambda_NCE > 0.0:
                self.optimizer_F = torch.optim.Adam(
                    itertools.chain(self.netF1.parameters(), self.netF2.parameters(), self.netF3.parameters(),
                                    self.netF4.parameters(),
                                    self.netF5.parameters(), self.netF6.parameters()), lr=self.cfg.lr,
                    betas=(self.cfg.beta1, self.cfg.beta2))
                self.optimizers.append(self.optimizer_F)

    def optimize_parameters(self):
        # forward
        self.forward()

        # update D
        self.set_requires_grad([self.netD_A, self.netD_B], True)
        self.optimizer_D.zero_grad()
        self.backward_D_A()  # calculate gradients for D_A
        self.backward_D_B()  # calculate graidents for D_B
        self.optimizer_D.step()
        # update G
        self.set_requires_grad([self.netD_A, self.netD_B], False)
        self.optimizer_G.zero_grad()
        if self.cfg.netF == 'mlp_sample':
            self.optimizer_F.zero_grad()
        self.loss_G = self.compute_G_loss()
        self.loss_G.backward()
        self.optimizer_G.step()
        if self.cfg.netF == 'mlp_sample':
            self.optimizer_F.step()

    def set_input(self, input):
        """Unpack input data from the dataloader and perform necessary pre-processing steps.
        Parameters:
            input (dict): include the data itself and its metadata information.
        The option 'direction' can be used to swap domain A and domain B.
        """
        AtoB = self.cfg.direction == 'AtoB'
        self.real_A = input['A' if AtoB else 'B'].to(self.device)
        self.real_B = input['B' if AtoB else 'A'].to(self.device)
        self.image_paths = input['A_paths' if AtoB else 'B_paths']

    def forward(self):
        """Run forward pass; called by both functions <optimize_parameters> and <test>."""
        self.fake_B = self.netG_A(self.real_A)  # G_A(A)
        self.fake_A = self.netG_B(self.real_B)  # G_B(B)

        if self.cfg.nce_idt:
            self.idt_A = self.netG_A(self.real_B)
            self.idt_B = self.netG_B(self.real_A)

    def backward_D_basic(self, netD, real, fake):
        """Calculate GAN loss for the discriminator
        """
        # Real
        pred_real = netD(real)
        loss_D_real = self.criterionGAN(pred_real, True)
        # Fake
        pred_fake = netD(fake.detach())
        loss_D_fake = self.criterionGAN(pred_fake, False)
        # Combined loss and calculate gradients
        loss_D = (loss_D_real + loss_D_fake) * 0.5
        loss_D.backward()
        return loss_D

    def backward_D_A(self):
        """Calculate GAN loss for discriminator D_A"""
        fake_B = self.fake_B_pool.query(self.fake_B)
        self.loss_D_A = self.backward_D_basic(self.netD_A, self.real_B, fake_B)

    def backward_D_B(self):
        """Calculate GAN loss for discriminator D_B"""
        fake_A = self.fake_A_pool.query(self.fake_A)
        self.loss_D_B = self.backward_D_basic(self.netD_B, self.real_A, fake_A)

    def compute_G_loss(self):
        """Calculate GAN and NCE loss for the generator"""
        fakeB = self.fake_B
        fakeA = self.fake_A

        # First, G(A) should fake the discriminator
        if self.cfg.lambda_GAN > 0.0:
            pred_fakeB = self.netD_A(fakeB)
            pred_fakeA = self.netD_B(fakeA)
            self.loss_G_A = self.criterionGAN(pred_fakeB, True).mean() * self.cfg.lambda_GAN
            self.loss_G_B = self.criterionGAN(pred_fakeA, True).mean() * self.cfg.lambda_GAN
        else:
            self.loss_G_A = 0.0
            self.loss_G_B = 0.0
        # L1 IDENTICAL LOSS
        self.loss_idt_A = self.criterionIdt(self.idt_A, self.real_B)
        self.loss_idt_B = self.criterionIdt(self.idt_B, self.real_A)
        # Similarity Loss and NCE losses
        self.loss_Sim, self.loss_NCE1, self.loss_NCE2 = self.calculate_Sim_loss_all \
            (self.real_A, self.fake_B, self.real_B, self.fake_A)
        loss_NCE_both = (self.loss_NCE1 + self.loss_NCE2) * 0.5 + (self.loss_idt_A + self.loss_idt_B) * 0.5 \
                        + self.loss_Sim
        self.loss_G = (self.loss_G_A + self.loss_G_B) * 0.5 + loss_NCE_both
        return self.loss_G

    def calculate_Sim_loss_all(self, src1, tgt1, src2, tgt2):
        n_layers = len(self.nce_layers)
        feat_q1 = self.netG_B(tgt1, self.nce_layers)
        feat_k1 = self.netG_A(src1, self.nce_layers)
        feat_q2 = self.netG_A(tgt2, self.nce_layers)
        feat_k2 = self.netG_B(src2, self.nce_layers)
        feat_k_pool1, sample_ids1 = self.netF1(feat_k1, self.cfg.num_patches, None)
        feat_q_pool1, _ = self.netF2(feat_q1, self.cfg.num_patches, sample_ids1)
        feat_q_pool1_noid, _ = self.netF2(feat_q1, self.cfg.num_patches, None)
        feat_k_pool2, sample_ids2 = self.netF2(feat_k2, self.cfg.num_patches, None)
        feat_q_pool2, _ = self.netF1(feat_q2, self.cfg.num_patches, sample_ids2)
        feat_q_pool2_noid, _ = self.netF1(feat_q2, self.cfg.num_patches, None)

        nce_loss1 = 0.0
        for f_q, f_k, crit in zip(feat_q_pool1, feat_k_pool1, self.criterionNCE):
            loss = crit(f_q, f_k)
            nce_loss1 += loss.mean()

        nce_loss2 = 0.0
        for f_q, f_k, crit in zip(feat_q_pool2, feat_k_pool2, self.criterionNCE):
            loss = crit(f_q, f_k)
            nce_loss2 += loss.mean()

        m, n = self.cfg.num_patches, self.cfg.netF_nc
        nce_loss1 = nce_loss1 / n_layers
        nce_loss2 = nce_loss2 / n_layers
        feature_realA = torch.zeros([n_layers, m, n])
        feature_fakeB = torch.zeros([n_layers, m, n])
        feature_realB = torch.zeros([n_layers, m, n])
        feature_fakeA = torch.zeros([n_layers, m, n])
        for i in range(n_layers):
            feature_realA[i] = feat_k_pool1[i]
            feature_fakeB[i] = feat_q_pool1_noid[i]
            feature_realB[i] = feat_k_pool2[i]
            feature_fakeA[i] = feat_q_pool2_noid[i]
        feature_realA_out = self.netF3(feature_realA.to(self.device))
        feature_fakeB_out = self.netF4(feature_fakeB.to(self.device))
        feature_realB_out = self.netF5(feature_realB.to(self.device))
        feature_fakeA_out = self.netF6(feature_fakeA.to(self.device))
        sim_loss = self.criterionSim(feature_realA_out, feature_fakeA_out) + \
                   self.criterionSim(feature_fakeB_out, feature_realB_out)

        return sim_loss * self.cfg.lambda_SIM, nce_loss1, nce_loss2

    def generate_visuals_for_evaluation(self, data, mode):
        with torch.no_grad():
            visuals = {}
            AtoB = self.cfg.direction == "AtoB"
            G = self.netG_A
            source = data["A" if AtoB else "B"].to(self.device)
            if mode == "forward":
                visuals["fake_B"] = G(source)
            else:
                raise ValueError("mode %s is not recognized" % mode)
            return visuals