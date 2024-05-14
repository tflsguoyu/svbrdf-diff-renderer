# -*- coding: utf-8 -*-
#
# Copyright (c) 2023, Yu Guo. All rights reserved.

import json
import tqdm
import numpy as np
import torch as th
from datetime import datetime

from .imageio import imread, imwrite, img9to1, tex4to1
from .optimization import Optim


class SvbrdfOptim(Optim):
    def __init__(self, device, renderer_obj):
        super().__init__(device, renderer_obj)
        self.res = renderer_obj.res

    def init_from_tex(self, textures):
        self.textures = self.gradient(textures)

    def init_from_const(self, dif=0.5, spe=0.04, rgh=0.2):
        normal_th = th.zeros(1, 2, self.res, self.res, device=self.device)
        diffuse_th = th.ones(1, 3, self.res, self.res, device=self.device) * dif * 2 - 1
        specular_th = th.ones(1, 3, self.res, self.res, device=self.device) * spe * 2 - 1
        roughness_th = th.ones(1, 1, self.res, self.res, device=self.device) * rgh * 2 - 1

        textures = th.cat((diffuse_th, normal_th, roughness_th, specular_th), 1)
        self.textures = self.gradient(textures)

    def init_from_randn(self, dif=0.5, spe=0.04, rgh=0.2):
        normal_th = (th.randn(1, 2, self.res, self.res, device=self.device) / 4).clamp(-1, 1)
        diffuse_th = (th.randn(1, 3, self.res, self.res, device=self.device) / 8 + dif).clamp(0, 1) * 2 - 1
        specular_th = (th.randn(1, 3, self.res, self.res, device=self.device) / 32 + spe).clamp(0, 1) * 2 - 1
        roughness_th = (th.randn(1, 1, self.res, self.res, device=self.device) / 16 + rgh).clamp(0, 1) * 2 - 1

        textures = th.cat((diffuse_th, normal_th, roughness_th, specular_th), 1)
        self.textures = self.gradient(textures)

    def load_targets(self, targets):
        self.targets = targets

    def optim(self, epochs, lr, svbrdf_obj):
        tmp_dir = svbrdf_obj.optimize_dir / "tmp" / str(datetime.now()).replace(" ", "-").replace(":", "-").replace(".", "-")
        tmp_dir.mkdir(parents=True, exist_ok=True)

        self.optimizer = th.optim.Adam([self.textures], lr=lr, betas=(0.9, 0.999))

        loss_image_list = []
        pbar = tqdm.trange(epochs)
        for epoch in pbar:
            # compute renderings
            rendereds = self.renderer_obj.eval(self.textures.clamp(-1, 1))

            # compute loss
            loss = self.compute_image_loss(rendereds)
            loss_image_list.append(loss.item())

            pbar.set_postfix({"Loss": loss.item()})

            # optimize
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            # save process
            if (epoch + 1) % 100 == 0 or epoch == 0 or epoch == (epochs - 1):
                tmp_this_dir = tmp_dir / f"{epoch + 1}"
                tmp_this_dir.mkdir(parents=True, exist_ok=True)

                self.save_loss([loss_image_list], ["image loss"], tmp_dir / "loss.jpg", epochs)

                svbrdf_obj.save_textures_th(self.textures.clamp(-1, 1), tmp_this_dir)

                rendereds = self.renderer_obj.eval(self.textures.clamp(-1, 1))
                svbrdf_obj.save_images_th(rendereds, tmp_this_dir)


class SvbrdfIO:
    def __init__(self, json_dir, device):
        self.device = device

        if not json_dir.exists():
            print(f"[ERROR:SvbrdfIO:init] {json_dir} is not exists")
            exit()

        with open(json_dir, "r") as f:
            data = json.load(f)

        if "reference_dir" in data:
            self.reference_dir = json_dir.parent / data["reference_dir"]
        if "target_dir" in data:
            self.target_dir = json_dir.parent / data["target_dir"]
        if "optimize_dir" in data:
            self.optimize_dir = json_dir.parent / data["optimize_dir"]
        if "rerender_dir" in data:
            self.rerender_dir = json_dir.parent / data["rerender_dir"]
        if "im_size" in data:
            self.im_size = data["im_size"]
        if "idx" in data:
            self.idx = data["idx"]
            self.n_of_imgs = len(self.idx)
        if "camera_pos" in data:
            self.camera_pos = data["camera_pos"]
        if "light_pos" in data:
            self.light_pos = data["light_pos"]
        if "light_pow" in data:
            self.light_pow = data["light_pow"]
            self.load_calibration_th()

        print("[DONE:SvbrdfIO] Initial object")

    def np_to_th(self, arr):
        return th.from_numpy(arr).to(self.device)

    def th_to_np(self, arr):
        return arr.detach().cpu().numpy()

    def reconstruct_normal(self, texture):
        normal_x  = texture[:, 0, :, :].clamp(-1, 1)
        normal_y  = texture[:, 1, :, :].clamp(-1, 1)
        normal_xy = (normal_x**2 + normal_y**2).clamp(0, 1)
        normal_z  = (1 - normal_xy).sqrt()
        normal    = th.stack((normal_x, normal_y, normal_z), 1)
        return normal / (normal.norm(2.0, 1, keepdim=True))

    def load_calibration_th(self):
        camera_pos = np.array(self.camera_pos, "float32")
        light_pos = np.array(self.light_pos, "float32")
        light_pow = np.array(self.light_pow, "float32")

        camera_pos = camera_pos[self.idx, :]
        light_pos = light_pos[self.idx, :]

        camera_pos_th = self.np_to_th(camera_pos)
        light_pos_th = self.np_to_th(light_pos)
        light_pow_th = self.np_to_th(light_pow)

        self.cl = [camera_pos_th, light_pos_th, light_pow_th]

        print("[DONE:SvbrdfIO] Load parameters")

    def load_textures_th(self, textures_dir, res):
        if not textures_dir.exists:
            print(f"[ERROR:SvbrdfIO:load_textures_th] {textures_dir} is not exists")
            exit()

        normal = imread(textures_dir / "nom.png", "normal", (res, res))
        diffuse = imread(textures_dir / "dif.png", "srgb", (res, res))
        specular = imread(textures_dir / "spe.png", "srgb", (res, res))
        roughness = imread(textures_dir / "rgh.png", "rough", (res, res))

        normal_th = self.np_to_th(normal).permute(2, 0, 1).unsqueeze(0)
        diffuse_th = self.np_to_th(diffuse*2-1).permute(2, 0, 1).unsqueeze(0)
        specular_th = self.np_to_th(specular*2-1).permute(2, 0, 1).unsqueeze(0)
        roughness_th = self.np_to_th(roughness*2-1).unsqueeze(0).unsqueeze(0)

        print("[DONE:SvbrdfIO] Load textures (numbers in range [-1,1])")
        return th.cat((diffuse_th, normal_th[:, :2, :, :], roughness_th, specular_th), 1)

    def save_textures_th(self, textures_th, textures_dir):
        textures_dir.mkdir(parents=True, exist_ok=True)

        diffuse_th = (textures_th[:, 0:3, :, :] + 1) / 2
        normal_th = textures_th[:, 3:5, :, :]
        roughness_th = (textures_th[:, 5, :, :] + 1) / 2
        specular_th = (textures_th[:, 6:9, :, :] + 1) / 2
        normal_th = self.reconstruct_normal(normal_th)

        normal = self.th_to_np(normal_th.squeeze().permute(1, 2, 0))
        diffuse = self.th_to_np(diffuse_th.squeeze().permute(1, 2, 0))
        specular = self.th_to_np(specular_th.squeeze().permute(1, 2, 0))
        roughness = self.th_to_np(roughness_th.squeeze())

        imwrite(normal, textures_dir / "nom.png", "normal")
        imwrite(diffuse, textures_dir / "dif.png", "srgb")
        imwrite(specular, textures_dir / "spe.png", "srgb")
        imwrite(roughness, textures_dir / "rgh.png", "rough")

        tex4to1(textures_dir)

        print("[DONE:SvbrdfIO] Save textures")

    def load_images_th(self, images_dir, res=256):
        if not images_dir.exists:
            print(f"[ERROR:SvbrdfIO:load_images_th] {images_dir} is not exists")
            exit()

        images = np.zeros((self.n_of_imgs, 3, res, res), dtype="float32")
        images_th = self.np_to_th(images)
        for i, idx in enumerate(self.idx):
            fn_image = images_dir / f"{idx:02d}.png"
            image = imread(fn_image, "srgb", (res, res))
            images_th[i, :, :, :] = self.np_to_th(image).permute(2, 0, 1)

        print("[DONE:SvbrdfIO] Load images")
        return images_th

    def save_images_th(self, images_th, images_dir):
        images_dir.mkdir(parents=True, exist_ok=True)

        if images_th.shape[0] != self.n_of_imgs:
            print("[ERROR:SvbrdfIO:save_images_th]")
            exit()

        for i, idx in enumerate(self.idx):
            image = self.th_to_np(images_th[i, :, :, :].permute(1, 2, 0))
            fn_image = images_dir / f"{idx:02d}.png"
            imwrite(image, fn_image, "srgb")

        if self.n_of_imgs == 9:
            img9to1(images_dir)

        print("[DONE:SvbrdfIO] Save images")
