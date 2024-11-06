# -*- coding: utf-8 -*-
#
# Copyright (c) 2024, Yu Guo. All rights reserved.

import os
import numpy as np
import torch as th
import tqdm
import requests
from pathlib import Path
from datetime import datetime

from .descriptor import VGGLoss
from .globalvar import init_global_noise
from . import globalvar
from .optimization import Optim
from .higan_models.stylegan2_generator import StyleGAN2Generator
# We use the generator of StyleGAN2 from higan (https://github.com/genforce/higan)


class MaterialGANOptim(Optim):
    def __init__(self, device, renderer_obj, ckp):
        super().__init__(device, renderer_obj)

        self.init_download_ckp()

        self.net_obj = StyleGAN2Generator('MaterialGAN', ckp)

        self.loss_large_feature = VGGLoss(device, np.array([1, 1, 4, 8])/14)
        for p in self.loss_large_feature.parameters():
            p.requires_grad = False

        self.loss_small_feature = VGGLoss(device, np.array([8, 8, 2, 1])/19)
        for p in self.loss_small_feature.parameters():
            p.requires_grad = False

    def init_download_ckp(self):
        ckp = "ckp/materialgan.pth"
        if not os.path.exists(ckp):
            url = "https://huggingface.co/tflsguoyu/MaterialGAN/resolve/main/materialgan.pth"
            self._download_checkpoint(ckp, url)        

        ckp = "ckp/latent_avg_W+_256.pt"
        if not os.path.exists(ckp):
            url = "https://huggingface.co/tflsguoyu/MaterialGAN/resolve/main/latent_avg_W+_256.pt"
            self._download_checkpoint(ckp, url)        

        ckp = "ckp/latent_const_N_256.pt"
        if not os.path.exists(ckp):
            url = "https://huggingface.co/tflsguoyu/MaterialGAN/resolve/main/latent_const_N_256.pt"
            self._download_checkpoint(ckp, url)        

        ckp = "ckp/latent_const_W+_256.pt"
        if not os.path.exists(ckp):
            url = "https://huggingface.co/tflsguoyu/MaterialGAN/resolve/main/latent_const_W+_256.pt"
            self._download_checkpoint(ckp, url)        

    def init_from(self, ckp):
        # initialize latent W+
        if len(ckp) == 0:
            latent_z = th.randn(1, 512).to(self.device)
            latent_w = self.net_obj.net.mapping(latent_z)
            latent = self.net_obj.net.truncation(latent_w)
        else:
            latent = th.load(ckp[0], map_location=self.device, weights_only=True)
        self.latent = self.gradient(latent)

        # initialize noise
        # return gloval var "noises"
        if len(ckp) == 2:
            init_global_noise(self.device, init_from=ckp[1])
        elif len(ckp) == 1:
            init_global_noise(self.device, init_from="avg")
        else:
            init_global_noise(self.device, init_from="random")

    def load_targets(self, targets):
        self.targets = targets
        self.loss_large_feature.load(targets)
        self.loss_small_feature.load(targets)

    def compute_feature_loss(self, predicts, flag):
        if flag == "L":
            return self.loss_large_feature(predicts)
        elif flag == "N":
            return self.loss_small_feature(predicts)
        else:
            print("[ERROR:MaterialGANOptim] To compute feature loss, a flag is needed, 'L' or 'N'")
            exit()

    def latent_to_textures(self, latent):
        textures_tmp = self.net_obj.net.synthesis(latent)
        # Option 1:
        # self.textures = textures.clamp(-1,1)
        # Option 2:
        textures = textures_tmp.clone()
        textures[:, 0:5, :, :] = textures_tmp[:, 0:5, :, :].clamp(-1, 1)
        textures[:,   5, :, :] = textures_tmp[:,   5, :, :].clamp(-0.5, 0.5)
        textures[:, 6:9, :, :] = textures_tmp[:, 6:9, :, :].clamp(-1, 1)
        return textures

    def optim(self, epochs, lr, svbrdf_obj, optim_light):
        tmp_name = str(datetime.now()).replace(" ", "-").replace(":", "-").replace(".", "-")
        tmp_dir = svbrdf_obj.optimize_dir / "tmp" / tmp_name
        tmp_dir.mkdir(parents=True, exist_ok=True)

        total_epochs, l_epochs, n_epochs = epochs
        cycle_epochs = l_epochs + n_epochs

        if optim_light:
            svbrdf_obj.cl[2] = self.gradient(svbrdf_obj.cl[2])

        loss_image_list = []
        loss_feature_list = []
        pbar = tqdm.trange(total_epochs)
        for epoch in pbar:
            # choose which variables to optimize
            epoch_tmp = epoch % cycle_epochs
            if int(epoch_tmp / l_epochs) == 0:  # optimize latent w+
                if optim_light:
                    self.optimizer = th.optim.Adam([self.latent] + [svbrdf_obj.cl[2]], lr=lr, betas=(0.9, 0.999))
                else:    
                    self.optimizer = th.optim.Adam([self.latent], lr=lr, betas=(0.9, 0.999))
                which_to_optimize = "L"
            else:  # optimize nosie
                if optim_light:
                    self.optimizer = th.optim.Adam(globalvar.noises + [svbrdf_obj.cl[2]], lr=lr, betas=(0.9, 0.999))
                else:
                    self.optimizer = th.optim.Adam(globalvar.noises, lr=lr, betas=(0.9, 0.999))
                which_to_optimize = "N"

            # compute renderings
            if optim_light:
                self.renderer_obj.update_light(svbrdf_obj.cl[2])
            textures = self.latent_to_textures(self.latent)
            rendereds = self.renderer_obj.eval(textures)

            # compute loss
            loss = 0
            loss_image = self.compute_image_loss(rendereds)
            loss_image_list.append(loss_image.item())
            loss += loss_image

            loss_feature = self.compute_feature_loss(rendereds, which_to_optimize) * 0.1
            loss_feature_list.append(loss_feature.item())
            loss += loss_feature

            pbar.set_postfix({"Loss": loss.item(), "Light": [int(svbrdf_obj.cl[2][0].item()), int(svbrdf_obj.cl[2][1].item()), int(svbrdf_obj.cl[2][2].item())]})

            # optimize
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            # save process
            if (epoch + 1) % 200 == 0 or epoch == 0 or epoch == (total_epochs - 1):
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
        self.loss = loss.item()
        self.loss_image = loss_image.item()

    def _download_checkpoint(self, ckp_path, url):
        """Download the checkpoint if it doesn't exist."""
        # Create directory if it doesn't exist
        Path(ckp_path).parent.mkdir(parents=True, exist_ok=True)
        
        try:
            response = requests.get(url, stream=True)
            response.raise_for_status()
            
            # Get total file size
            total_size = int(response.headers.get('content-length', 0))
            block_size = 1024  # 1 KB

            # Download with progress bar
            with open(ckp_path, 'wb') as f, tqdm.tqdm(
                desc="Downloading checkpoint",
                total=total_size,
                unit='iB',
                unit_scale=True,
                unit_divisor=1024,
            ) as pbar:
                for data in response.iter_content(block_size):
                    size = f.write(data)
                    pbar.update(size)
                    
            print(f"Successfully downloaded checkpoint to {ckp_path}")
            
        except Exception as e:
            print(f"Error downloading checkpoint: {e}")
            if os.path.exists(ckp_path):
                os.remove(ckp_path)  # Remove partial download
            raise