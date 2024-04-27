# -*- coding: utf-8 -*-
#
# Copyright (c) 2024, Yu Guo. All rights reserved.

import os
import torch as th
import tqdm
from datetime import datetime

from .optimization import Optim
from .higan_models.stylegan2_generator import StyleGAN2Generator
# We use the generator of StyleGAN2 from higan (https://github.com/genforce/higan)
from .globalvar import init_global_noise
from . import globalvar


class MaterialGANOptim(Optim):

    def __init__(self, device, renderer_obj, ckp):
        super().__init__(device, renderer_obj)
        self.net_obj = StyleGAN2Generator('MaterialGAN', ckp)

    def init_from(self, ckp):
        # initialize latent W+
        if len(ckp) == 0:
            latent_z = th.randn(1,512).to(self.device)
            latent_w = self.net_obj.net.mapping(latent_z)
            latent = self.net_obj.net.truncation(latent_w)
        else:
            if os.path.exists(ckp[0]):
                latent = th.load(ckp[0], map_location=self.device)
            else:
                print("[ERROR:MaterialGANOptim] Can not find latent vector ", ckp)
                exit()
        self.latent = self.gradient(latent)

        # initialize noise
        # return gloval var "noises"
        if len(ckp) == 2:
            init_global_noise(self.device, init_from=ckp[1])
        else:
            init_global_noise(self.device, init_from="random")

    def latent_to_textures(self):
        textures = self.net_obj.net.synthesis(self.latent)
        # Option 1: 
        # self.textures = textures.clamp(-1,1)
        # Option 2:
        self.textures = textures.clone()
        self.textures[:,0:5,:,:] = textures[:,0:5,:,:].clamp(-1,1)
        self.textures[:,5,:,:] = textures[:,5,:,:].clamp(-0.3,0.5)
        self.textures[:,6:9,:,:] = textures[:,6:9,:,:].clamp(-1,1)

    def optim(self, epochs, lr, svbrdf_obj):
        tmp_dir = svbrdf_obj.optimize_dir / "tmp" / str(datetime.now())
        tmp_dir.mkdir(parents=True, exist_ok=True)

        total_epochs, l_epochs, n_epochs = epochs
        cycle_epochs = l_epochs + n_epochs
        
        loss_image_list = []
        loss_lpips_list = []
        loss_feature_list = []
        pbar = tqdm.trange(total_epochs)
        for epoch in pbar:
            # choose which variables to optimize
            epoch_tmp = epoch % cycle_epochs
            if int(epoch_tmp / l_epochs) == 0:  # optimize latent w+
                self.optimizer = th.optim.Adam([self.latent], lr=lr, betas=(0.9, 0.999))
                which_to_optimize = "L"
            else:  # optimize nosie    
                self.optimizer = th.optim.Adam(globalvar.noises, lr=lr, betas=(0.9, 0.999))
                which_to_optimize = "N"

            # compute renderings
            self.latent_to_textures()
            rendereds = self.renderer_obj.eval(self.textures)

            # compute loss
            loss = 0
            loss_image = self.compute_image_loss(rendereds)
            loss_image_list.append(loss_image.item())
            loss += loss_image

            # loss_lpips = self.compute_lpips_loss(rendereds)
            # loss_lpips_list.append(loss_lpips.item())
            # loss += loss_lpips

            loss_feature = self.compute_feature_loss(rendereds, which_to_optimize) * 10
            loss_feature_list.append(loss_feature.item())
            loss += loss_feature

            pbar.set_postfix({"Loss": loss.item()})

            # optimize
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            # save process
            if (epoch + 1) % 100 == 0 or epoch == 0:
                tmp_this_dir = tmp_dir / f"{epoch}"
                tmp_this_dir.mkdir(parents=True, exist_ok=True)
                svbrdf_obj.save_textures_th(self.textures, tmp_this_dir)
                self.save_loss([loss_image_list, loss_feature_list], ["image loss", "feature loss"], tmp_dir / "loss.jpg", total_epochs)

        self.latent_to_textures()
