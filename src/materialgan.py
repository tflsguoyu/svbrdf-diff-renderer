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

    def latent_to_textures(self, latent):
        textures_tmp = self.net_obj.net.synthesis(latent)
        # Option 1: 
        # self.textures = textures.clamp(-1,1)
        # Option 2:
        textures = textures_tmp.clone()
        textures[:,0:5,:,:] = textures_tmp[:,0:5,:,:].clamp(-1,1)
        textures[:,5,:,:] = textures_tmp[:,5,:,:].clamp(-0.3,1)
        textures[:,6:9,:,:] = textures_tmp[:,6:9,:,:].clamp(-1,1)
        return textures

    def optim(self, epochs, lr, svbrdf_obj):
        tmp_dir = svbrdf_obj.optimize_dir / "tmp" / str(datetime.now()).replace(" ", "-").replace(":", "-").replace(".", "-")
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
            textures = self.latent_to_textures(self.latent)
            rendereds = self.renderer_obj.eval(textures)

            # compute loss
            loss = 0
            loss_image = self.compute_image_loss(rendereds) * 1000
            loss_image_list.append(loss_image.item())
            loss += loss_image

            # loss_lpips = self.compute_lpips_loss(rendereds)
            # loss_lpips_list.append(loss_lpips.item())
            # loss += loss_lpips

            loss_feature = self.compute_feature_loss(rendereds, which_to_optimize) * 0.001
            loss_feature_list.append(loss_feature.item())
            loss += loss_feature

            pbar.set_postfix({"Loss": loss.item()})

            # optimize
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            # save process
            if (epoch + 1) % 100 == 0 or epoch == 0 or epoch == (total_epochs - 1):
                tmp_this_dir = tmp_dir / f"{epoch + 1}"
                tmp_this_dir.mkdir(parents=True, exist_ok=True)

                self.save_loss([loss_image_list, loss_feature_list], ["image loss", "feature loss"], tmp_dir / "loss.jpg", total_epochs)

                th.save(self.latent, tmp_this_dir / "optim_latent.pt")
                th.save(globalvar.noises, tmp_this_dir / "optim_noise.pt")

                textures = self.latent_to_textures(self.latent)
                svbrdf_obj.save_textures_th(textures, tmp_this_dir)

                rendereds = self.renderer_obj.eval(textures)
                svbrdf_obj.save_images_th(rendereds, tmp_this_dir)

        self.textures = textures