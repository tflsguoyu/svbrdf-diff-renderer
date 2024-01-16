# -*- coding: utf-8 -*-
#
# Copyright (c) 2024, Yu Guo. All rights reserved.

import os
import torch as th

from .optimization import Optim
from .higan_models.stylegan2_generator import StyleGAN2Generator
# We use the generator of StyleGAN2 from higan (https://github.com/genforce/higan)

class MaterialGANOptim(Optim):

    def __init__(self, device, ckp, loss_type="L2"):
        super().__init__(device, loss_type)
        self.net_obj = StyleGAN2Generator('MaterialGAN', ckp)

    def init_latent(self, latent_type="w+", init_from="random"):
        if init_from == "random":
            if latent_type == "z":
                self.latent = th.randn(1,512).to(self.device)
            elif latent_type == "w":
                self.latent = th.randn(1,512).to(self.device)
                self.latent = self.net_obj.net.mapping(self.latent)
            elif latent_type == 'w+':
                self.latent = th.randn(1,512).to(self.device)
                self.latent = self.net_obj.net.mapping(self.latent)
                self.latent = self.net_obj.net.truncation(self.latent)
            else:
                print("[ERROR:MaterialGANOptim] latent_type should be z|w|w+")
                exit()
        else:
            if os.path.exists(init_from):
                self.latent = th.load(init_from).to(self.device)
            else:
                print("[ERROR:MaterialGANOptim] Can not find latent vector ", init_from)
                exit()

    def latent_to_textures(self, latent_type="w+"):
        if latent_type == 'z':
            self.latent = self.net_obj.net.mapping(self.latent)
            self.latent = self.net_obj.net.truncation(self.latent)
        elif latent_type == 'w':
            self.latent = self.net_obj.net.truncation(self.latent)
        elif latent_type == 'w+':
            pass
        else:
            print("[ERROR:MaterialGANOptim] latent_to_textures should be z|w|w+")
            exit()

        net_out = self.net_obj.net.synthesis(self.latent)
        net_out = net_out.clamp(-1,1)

        self.textures = self.netout_to_textures(net_out)

    def netout_to_textures(self, netout):
        diffuse_th = (netout[:, 0:3, :, :] + 1) * 0.5
        normal_th = netout[:, 3:5, :, :]
        roughness_th = (netout[:, 5, :, :] + 1) * 0.25 + 0.2 # convert to [0.2, 0.7]
        specular_th = (netout[:, 6:9, :, :] + 1) * 0.5
        return [normal_th, diffuse_th, specular_th, roughness_th]


    def optim(self, epochs, lr=0.01):
        self.optimizer = th.optim.Adam(self.textures, lr=lr, betas=(0.9, 0.999))
        self.iteration(epochs)
