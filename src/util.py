# -*- coding: utf-8 -*-
#
# Copyright (c) 2023, Yu Guo. All rights reserved.

import json
import numpy as np
import torch as th
from pathlib import Path
from src.imageio import imread, imwrite


class SvbrdfIO:
    def __init__(self, folder, device):
        self.folder = folder
        self.device = device

        json_dir = folder / "parameters.json"
        if not json_dir.exists():
            print(f"[ERROR:SvbrdfIO:init] {json_dir} is not exists")
            exit()

        f = open(json_dir)
        data = json.load(f)
        f.close()

        self.textures_dir = data["textures_dir"]
        self.images_dir = data["images_dir"]
        self.im_size = data["image_size"]
        self.index = data["id"]
        self.camera_pos = data["camera_pos"]
        self.light_pos = data["light_pos"]
        self.light_pow = data["light_pow"]

        self.n_of_imgs = len(self.index)

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


    def load_parameters_th(self):
        camera_pos = np.array(self.camera_pos, 'float32')
        light_pos = np.array(self.light_pos, 'float32')
        light_pow = np.array(self.light_pow, 'float32')

        camera_pos = camera_pos[self.index, :]
        light_pos = light_pos[self.index, :]

        camera_pos_th = self.np_to_th(camera_pos)
        light_pos_th = self.np_to_th(light_pos)
        light_pow_th = self.np_to_th(light_pow)

        print("[DONE:SvbrdfIO] load parameters")
        return camera_pos_th, light_pos_th, light_pow_th


    def load_textures_th(self, opt_ref, res):
        # opt_ref is either str("optimized") or str("reference")
        textures_dir = self.folder / self.textures_dir / f"{opt_ref}/{res}"
        if not textures_dir.exists:
            print(f"[ERROR:SvbrdfIO:load_textures_th] {textures_dir} is not exists")
            exit()

        normal = imread(textures_dir / "nom.png", 'normal')
        diffuse = imread(textures_dir / "dif.png", 'srgb')
        specular = imread(textures_dir / "spe.png", 'srgb')
        roughness = imread(textures_dir / "rgh.png", 'rough')

        if normal.shape[0] != res or normal.shape[1] != res:
            print(f"[ERROR:SvbrdfIO:load_textures_th] textures in {textures_dir} have wrong resolution")
            exit()

        normal_th = self.np_to_th(normal).permute(2, 0, 1).unsqueeze(0)
        diffuse_th = self.np_to_th(diffuse).permute(2, 0, 1).unsqueeze(0)
        specular_th = self.np_to_th(specular).permute(2, 0, 1).unsqueeze(0)
        roughness_th = self.np_to_th(roughness).unsqueeze(0).unsqueeze(0)

        print("[DONE:SvbrdfIO] load textures")
        return normal_th[:, :2, :, :], diffuse_th, specular_th, roughness_th


    def save_textures_th(self, textures_th, opt_ref, res):
        # opt_ref is either str("optimized") or str("reference")
        textures_dir = self.folder / self.textures_dir / f"{opt_ref}/{res}"
        textures_dir.mkdir(parents=True, exists_ok=True)

        normal_th, diffuse_th, specular_th, roughness_th = textures_th
        normal_th = self.reconstruct_normal(normal_th)

        normal = self.th_to_np(normal_th.squeeze().permute(1, 2, 0))
        diffuse = self.th_to_np(diffuse_th.squeeze().permute(1, 2, 0))
        specular = self.th_to_np(specular_th.squeeze().permute(1, 2, 0))
        roughness = self.th_to_np(roughness_th.squeeze())

        if normal.shape[0] != res or normal.shape[1] != res:
            print(f"[ERROR:SvbrdfIO:save_textures_th] textures in {textures_dir} have wrong resolution")
            exit()

        imwrite(normal, textures_dir / 'nom.png', 'normal')
        imwrite(diffuse, textures_dir / 'dif.png', 'srgb')
        imwrite(specular, textures_dir / 'spe.png', 'srgb')
        imwrite(roughness, textures_dir / 'rgh.png', 'rough')

        print("[DONE:SvbrdfIO] save textures")


def load_targets(folder, resolution, device):

    f = open(os.path.join(folder, 'par.json'))
    data = json.load(f)
    f.close()

    index = data["id"]
    target_folder = data["target_folder"]

    fn_target = os.path.join(folder, target_folder, '00.png')
    target0 = imread(fn_target, 'sRGB')

    assert(target0.shape[0] == target0.shape[1])
    res = target0.shape[0]

    n_of_imgs = len(index)

    if resolution < 0:
        resolution = res

    targets = th.zeros((n_of_imgs, 3, resolution, resolution), dtype=th.float32, device=device)
    for i, idx in enumerate(index):
        fn_target = os.path.join(folder, target_folder, '%02d.png' % idx)
        target = imread(fn_target, 'sRGB')
        if resolution > 0:
            target = cv2.resize(target, (resolution, resolution), interpolation=cv2.INTER_LANCZOS4)
        targets[i, :, :, :] = th.from_numpy(target).permute(2, 0, 1)

    return res, targets.to(device)



