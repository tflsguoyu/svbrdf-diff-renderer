# -*- coding: utf-8 -*-
#
# Copyright (c) 2023, Yu Guo. All rights reserved.

import json
import tqdm
import numpy as np
import torch as th
from src.imageio import imread, imwrite, img9to1, tex4to1

from src.optimization import Optim

class SvbrdfOptim(Optim):

    def __init__(self, device, loss_type="L2"):
        super().__init__(device, loss_type)

    def load_textures_from_tex(self, textures):
        self.textures = self.gradient(textures)

    def load_textures_from_const(self, res):
        normal_th = th.zeros(1, 2, res, res, device=self.device)
        diffuse_th = th.ones(1, 3, res, res, device=self.device) * 0.5
        specular_th = th.ones(1, 3, res, res, device=self.device) * 0.04
        roughness_th = th.ones(1, 1, res, res, device=self.device) * 0.2

        textures = [normal_th, diffuse_th, specular_th, roughness_th]
        self.textures = self.gradient(textures)

    def load_textures_from_randn(self, res):
        normal_th = (th.randn(1, 2, res, res, device=self.device) / 4).clamp(-1, 1)
        diffuse_th = (th.randn(1, 3, res, res, device=self.device) / 8 + 0.5).clamp(0, 1)
        specular_th = (th.randn(1, 3, res, res, device=self.device) / 32 + 0.04).clamp(0, 1)
        roughness_th = (th.randn(1, 1, res, res, device=self.device) / 16 + 0.2).clamp(0, 1)

        textures = [normal_th, diffuse_th, specular_th, roughness_th]
        self.textures = self.gradient(textures)

    def optim(self, epochs, lr=0.01):
        self.optimizer = th.optim.Adam(self.textures, lr=lr, betas=(0.9, 0.999))
        self.iteration(epochs)


class SvbrdfIO:

    def __init__(self, folder, device):
        self.folder = folder
        self.device = device

        json_dir = folder / "parameters.json"
        if not json_dir.exists():
            print(f"[ERROR:SvbrdfIO:init] {json_dir} is not exists")
            exit()

        with open(json_dir, "r") as f:
            data = json.load(f)

        self.textures_dir = data["textures_dir"]
        self.images_dir = data["images_dir"]
        self.im_size = data["image_size"]
        self.index = data["idx"]
        self.camera_pos = data["camera_pos"]
        self.light_pos = data["light_pos"]
        self.light_pow = data["light_pow"]
        self.n_of_imgs = len(self.index)

        self.load_calibration_th()

        print("[DONE:SvbrdfIO] Initial object")


    def np_to_th(self, arr):
        return th.from_numpy(arr).to(self.device)


    def th_to_np(self, arr):
        return arr.detach().cpu().numpy()


    def reconstruct_normal(self, texture):
        normal_x  = texture[:,0,:,:].clamp(-1 ,1)
        normal_y  = texture[:,1,:,:].clamp(-1 ,1)
        normal_xy = (normal_x**2 + normal_y**2).clamp(0, 1)
        normal_z  = (1 - normal_xy).sqrt()
        normal    = th.stack((normal_x, normal_y, normal_z), 1)
        return normal / (normal.norm(2.0, 1, keepdim=True))


    def load_calibration_th(self):
        camera_pos = np.array(self.camera_pos, "float32")
        light_pos = np.array(self.light_pos, "float32")
        light_pow = np.array(self.light_pow, "float32")

        camera_pos = camera_pos[self.index, :]
        light_pos = light_pos[self.index, :]

        camera_pos_th = self.np_to_th(camera_pos)
        light_pos_th = self.np_to_th(light_pos)
        light_pow_th = self.np_to_th(light_pow)

        self.cl = [camera_pos_th, light_pos_th, light_pow_th]

        print("[DONE:SvbrdfIO] Load parameters")


    def load_textures_th(self, opt_ref, res):
        # opt_ref is either str("optimized") or str("reference")
        textures_dir = self.folder / self.textures_dir / f"{opt_ref}/{res}"
        if not textures_dir.exists:
            print(f"[ERROR:SvbrdfIO:load_textures_th] {textures_dir} is not exists")
            exit()

        normal = imread(textures_dir / "nom.png", "normal")
        diffuse = imread(textures_dir / "dif.png", "srgb")
        specular = imread(textures_dir / "spe.png", "srgb")
        roughness = imread(textures_dir / "rgh.png", "rough")

        if normal.shape[0] != res or normal.shape[1] != res:
            print(f"[ERROR:SvbrdfIO:load_textures_th] textures in {textures_dir} have wrong resolution")
            exit()

        normal_th = self.np_to_th(normal).permute(2, 0, 1).unsqueeze(0)
        diffuse_th = self.np_to_th(diffuse).permute(2, 0, 1).unsqueeze(0)
        specular_th = self.np_to_th(specular).permute(2, 0, 1).unsqueeze(0)
        roughness_th = self.np_to_th(roughness).unsqueeze(0).unsqueeze(0)

        print("[DONE:SvbrdfIO] Load textures")
        return [normal_th[:, :2, :, :], diffuse_th, specular_th, roughness_th]


    def save_textures_th(self, textures_th, opt_ref, res):
        # opt_ref is either str("optimized") or str("reference")
        textures_dir = self.folder / self.textures_dir / f"{opt_ref}/{res}"
        textures_dir.mkdir(parents=True, exist_ok=True)

        normal_th, diffuse_th, specular_th, roughness_th = textures_th
        normal_th = self.reconstruct_normal(normal_th)

        normal = self.th_to_np(normal_th.squeeze().permute(1, 2, 0))
        diffuse = self.th_to_np(diffuse_th.squeeze().permute(1, 2, 0))
        specular = self.th_to_np(specular_th.squeeze().permute(1, 2, 0))
        roughness = self.th_to_np(roughness_th.squeeze())

        if normal.shape[0] != res or normal.shape[1] != res:
            print(f"[ERROR:SvbrdfIO:save_textures_th] textures in {textures_dir} have wrong resolution")
            exit()

        imwrite(normal, textures_dir / "nom.png", "normal")
        imwrite(diffuse, textures_dir / "dif.png", "srgb")
        imwrite(specular, textures_dir / "spe.png", "srgb")
        imwrite(roughness, textures_dir / "rgh.png", "rough")

        tex4to1(textures_dir)

        print("[DONE:SvbrdfIO] Save textures")


    def load_images_th(self, opt_ref, res, res2=None):
        # opt_ref is either str("optimized") or str("reference")
        images_dir = self.folder / self.images_dir / f"{opt_ref}/{res}"
        if not images_dir.exists:
            print(f"[ERROR:SvbrdfIO:load_images_th] {images_dir} is not exists")
            exit()

        images = np.zeros((self.n_of_imgs, 3, res2, res2), dtype="float32")
        images_th = self.np_to_th(images)
        for i, idx in enumerate(self.index):
            fn_image = images_dir / f"{idx:02d}.png"
            image = imread(fn_image, "srgb", (res2, res2))
            images_th[i,:,:,:] = self.np_to_th(image).permute(2, 0, 1)

        print("[DONE:SvbrdfIO] Load images")
        return images_th


    def save_images_th(self, images_th, opt_ref, res):
        # opt_ref is either str("optimized") or str("reference")
        images_dir = self.folder / self.images_dir / f"{opt_ref}/{res}"
        images_dir.mkdir(parents=True, exist_ok=True)

        if images_th.shape[0] != self.n_of_imgs:
            print(f"[ERROR:SvbrdfIO:save_images_th]")
            exit()

        for i, idx in enumerate(self.index):
            image = self.th_to_np(images_th[i, :, :, :].permute(1, 2, 0))
            fn_image = images_dir / f"{idx:02d}.png"
            imwrite(image, fn_image, "srgb")

        if self.n_of_imgs == 9:
            img9to1(images_dir)

        print("[DONE:SvbrdfIO] Save images")

