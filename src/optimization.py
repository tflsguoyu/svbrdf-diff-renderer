# -*- coding: utf-8 -*-
#
# Copyright (c) 2024, Yu Guo. All rights reserved.

import torch as th
from lpips import LPIPS
from .descriptor import VGG19Loss

class Optim:

    def __init__(self, device, renderer_obj):
        self.device = device
        self.eps = 1e-4
        self.loss_l2 = th.nn.MSELoss().to(device)

        self.use_lpips = False
        self.use_vgg19 = False

        if self.use_lpips:
            self.loss_lpips = LPIPS(net='vgg').to(device)
            for p in self.loss_lpips.parameters():
                p.requires_grad = False

        if self.use_vgg19:
            self.loss_vgg19 = VGG19Loss(device)
            for p in self.loss_vgg19.parameters():
                p.requires_grad = False

        self.renderer_obj = renderer_obj

    def gradient(self, parameters):
        if isinstance(parameters, list):
            for idx, parameter in enumerate(parameters):
                parameters[idx] = th.autograd.Variable(parameter, requires_grad=True)
        else:
            parameters = th.autograd.Variable(parameters, requires_grad=True)
        return parameters

    def load_targets(self, targets):
        self.targets_srgb = self.srgb(targets)
        if self.use_vgg19:
            self.loss_vgg19.load(self.targets_srgb)
        self.res = targets.shape[-1]

    def srgb(self, images):
        return images.clamp(self.eps, 1) ** (1 / 2.2)

    def compute_image_loss(self, predicts):
        return self.loss_l2(self.srgb(predicts), self.targets_srgb)

    def compute_lpips_loss(self, predicts):
        return self.loss_lpips(self.srgb(predicts), self.targets_srgb, normalize=True).mean()

    def compute_vgg19_loss(self, predicts):
        return self.loss_vgg19(self.srgb(predicts))

    def optim(self, epochs, lr):
        raise NotImplementedError(f'Should be implemented in derived class!')
