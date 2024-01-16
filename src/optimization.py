# -*- coding: utf-8 -*-
#
# Copyright (c) 2024, Yu Guo. All rights reserved.

import tqdm
import torch as th
from lpips import LPIPS
from .descriptor import VGG19Loss

class Optim:

    def __init__(self, device, loss_type):
        self.device = device
        self.eps = 1e-4
        self.loss_type = loss_type
        self.loss_l2 = th.nn.MSELoss().to(device)

        if loss_type == "LPIPS":
            self.loss_lpips = LPIPS(net='vgg').to(device)
            for p in self.loss_lpips.parameters():
                p.requires_grad = False

        if loss_type == "VGG19":
            self.loss_vgg19 = VGG19Loss(device)
            for p in self.loss_vgg19.parameters():
                p.requires_grad = False

    def gradient(self, parameters):
        for idx, parameter in enumerate(parameters):
            parameters[idx] = th.autograd.Variable(parameter, requires_grad=True)
        return parameters

    def load_targets(self, images):
        self.targets_srgb = self.srgb(images)
        if self.loss_type == "VGG19":
            self.loss_vgg19.load(self.targets_srgb)

    def load_renderer(self, renderer):
        self.renderer_obj = renderer

    def srgb(self, images):
        return images.clamp(self.eps, 1) * (1 / 2.2)

    def compute_image_loss(self, predicts):
        return self.loss_l2(self.srgb(predicts), self.targets_srgb)

    def compute_lpips_loss(self, predicts):
        return self.loss_lpips(self.srgb(predicts), self.targets_srgb, normalize=True).mean()

    def compute_vgg19_loss(self, predicts):
        return self.loss_vgg19(self.srgb(predicts))

    def iteration(self, epochs):
        pbar = tqdm.trange(epochs)
        for epoch in pbar:
            # compute renderings
            rendereds = self.renderer_obj.eval(self.textures)

            # compute loss
            loss = self.compute_image_loss(rendereds)
            # loss = self.compute_lpips_loss(rendereds)
            # loss = self.compute_vgg19_loss(rendereds)
            pbar.set_postfix({"Loss": loss.item()})

            # optimize
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

    def optim(self, epochs, lr):
        raise NotImplementedError(f'Should be implemented in derived class!')
