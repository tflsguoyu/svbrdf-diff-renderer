# -*- coding: utf-8 -*-
#
# Copyright (c) 2024, Yu Guo. All rights reserved.

import numpy as np
import torch as th
import matplotlib.pyplot as plt
from lpips import LPIPS
from .descriptor import VGG19Loss

class Optim:

    def __init__(self, device, renderer_obj):
        self.device = device
        self.eps = 1e-4
        self.loss_l2 = th.nn.MSELoss().to(device)

        self.use_lpips = False
        self.use_vgg19 = True

        if self.use_lpips:
            self.loss_lpips = LPIPS(net='alex').to(device)
            for p in self.loss_lpips.parameters():
                p.requires_grad = False

        if self.use_vgg19:
            self.loss_large_feature = VGG19Loss(device, "1148")
            for p in self.loss_large_feature.parameters():
                p.requires_grad = False

            self.loss_small_feature = VGG19Loss(device, "8821")
            for p in self.loss_small_feature.parameters():
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
            self.loss_large_feature.load(self.targets_srgb)
            self.loss_small_feature.load(self.targets_srgb)
        self.res = targets.shape[-1]

    def srgb(self, images):
        return images.clamp(self.eps, 1) ** (1 / 2.2)

    def compute_image_loss(self, predicts):
        return self.loss_l2(self.srgb(predicts), self.targets_srgb)

    def compute_lpips_loss(self, predicts):
        return self.loss_lpips(self.srgb(predicts), self.targets_srgb, normalize=True).mean()

    def compute_feature_loss(self, predicts, flag):
        if flag == "L":
            return self.loss_large_feature(self.srgb(predicts))
        elif flag == "N":
            return self.loss_small_feature(self.srgb(predicts))
        else:
            exit()

    def optim(self, epochs, lr, svbrdf_obj):
        raise NotImplementedError(f'Should be implemented in derived class!')

    def save_loss(self, losses, labels, save_dir, N):
        plt.figure(figsize=(8,4))
        for i in range(len(losses)):
            plt.plot(np.log1p(losses[i]), label=labels[i])
        plt.xlim(0, N)
        plt.legend()
        plt.title("log(1+loss)")
        plt.savefig(save_dir)
        plt.close()


