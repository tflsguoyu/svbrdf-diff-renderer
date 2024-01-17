# -*- coding: utf-8 -*-
#
# Copyright (c) 2024, Yu Guo. All rights reserved.

import os
import torch as th
import tqdm

from .optimization import Optim
from .higan_models.stylegan2_generator import StyleGAN2Generator
# We use the generator of StyleGAN2 from higan (https://github.com/genforce/higan)

class MaterialGANOptim(Optim):

    def __init__(self, device, renderer_obj, ckp):
        super().__init__(device, renderer_obj)
        self.net_obj = StyleGAN2Generator('MaterialGAN', ckp)

    def init_from_latent(self, init_from="random"):
        if init_from == "random":
            latent_z = th.randn(1,512).to(self.device)
            latent_w = self.net_obj.net.mapping(latent_z)
            self.latent = self.net_obj.net.truncation(latent_w)
        else:
            if os.path.exists(init_from):
                self.latent = th.load(init_from).to(self.device)
            else:
                print("[ERROR:MaterialGANOptim] Can not find latent vector ", init_from)
                exit()

    def latent_to_textures(self, latent_type="w+"):
        textures = self.net_obj.net.synthesis(self.latent)
        self.textures = textures.clamp(-1,1)

    def optim(self, epochs, lr=0.01):
        self.latent = th.autograd.Variable(self.latent, requires_grad=True)
        self.optimizer = th.optim.Adam([self.latent], lr=lr, betas=(0.9, 0.999))
        self.iteration(epochs)

    def iteration(self, epochs):
        pbar = tqdm.trange(epochs)
        for epoch in pbar:
            # compute renderings
            self.latent_to_textures()
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

