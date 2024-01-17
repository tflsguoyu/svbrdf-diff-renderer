# -*- coding: utf-8 -*-
#
# Copyright (c) 2023, Yu Guo. All rights reserved.

import numpy as np
import torch as th

class Microfacet:

    def __init__(self, res, n, size, cl, device):
        # Verified
        self.n_of_imgs = n
        self.f0 = 0.04
        self.eps = 1e-6

        tmp = th.arange(res, dtype=th.float32, device=device)
        tmp = ((tmp + 0.5) / res - 0.5) * size
        x, y = th.meshgrid(tmp, tmp, indexing='xy')
        self.pos = th.stack((x, -y, th.zeros_like(x)), 2)
        self.pos = self.pos.permute(2, 0 ,1).unsqueeze(0).expand(n, -1, -1, -1)

        self.camera_pos = cl[0].unsqueeze(2).unsqueeze(3).expand(-1, -1, res, res)
        self.light_pos = cl[1].unsqueeze(2).unsqueeze(3).expand(-1, -1, res, res)
        self.light_pow = cl[2].unsqueeze(0).unsqueeze(2).unsqueeze(3).expand(n, -1, res, res)

        print("[DONE:Microfacet] Initial object")

    def GGX(self, cos_h, alpha):
        c2 = cos_h ** 2
        a2 = alpha ** 2
        den = c2 * a2 + (1 - c2)
        return a2 / (np.pi * den**2 + self.eps)

    def Beckmann(self, cos_h, alpha):
        c2 = cos_h ** 2
        t2 = (1 - c2) / c2
        a2 = alpha ** 2
        return th.exp(-t2 / a2) / (np.pi * a2 * c2 ** 2)

    def Fresnel_f0(self, cos, f0):
        return f0 + (1 - f0) * (1 - cos)**5

    def Fresnel(self, cos, specular):
        sphg = th.pow(2.0, ((-5.55473 * cos) - 6.98316) * cos);
        return specular + (1.0 - specular) * sphg

    def Smith(self, n_dot_v, n_dot_l, alpha):
        def _G1(cos, k):
            return cos / (cos * (1.0 - k) + k)

        k = alpha * 0.5 + self.eps
        return _G1(n_dot_v, k) * _G1(n_dot_l, k)

    def dot(self, a, b):
        return (a * b).sum(1, keepdim=True).expand(-1, 3, -1, -1)

    def normalize(self, vec):
        return vec / (vec.norm(2.0, 1, keepdim=True))

    def get_dir(self, pos):
        vec = pos - self.pos
        return self.normalize(vec), self.dot(vec, vec)

    def reconstruct_normal(self, texture):
        normal_x  = texture[:,0,:,:].clamp(-1 ,1)
        normal_y  = texture[:,1,:,:].clamp(-1 ,1)
        normal_xy = (normal_x**2 + normal_y**2).clamp(0, 1-self.eps)
        normal_z  = (1 - normal_xy).sqrt()  # The derivative of x.sqrt() when x=0 is nan
        normal    = th.stack((normal_x, normal_y, normal_z), 1)
        return self.normalize(normal)

    def tex2map(self, textures):
        # Reformats tensor from [1, 9, res, res] to four [N, 3, res, res] maps
        diffuse = ((textures[:, 0:3, :, :] + 1) / 2).expand(self.n_of_imgs, -1, -1, -1)
        normal = self.reconstruct_normal(textures[:, 3:5, :, :]).expand(self.n_of_imgs, -1, -1, -1)
        roughness = ((textures[:, 5, :, :] + 1) / 2).expand(self.n_of_imgs, 3, -1, -1)
        specular = ((textures[:, 6:9, :, :] + 1) / 2).expand(self.n_of_imgs, -1, -1, -1)

        return normal, diffuse, specular, roughness

    def eval(self, textures):

        normal, diffuse, specular, roughness = self.tex2map(textures)

        # Computes viewing direction, light direction, half angle
        v, _ = self.get_dir(self.camera_pos)
        l, dist_l_sq = self.get_dir(self.light_pos)
        h = self.normalize(l + v)

        # Computes dot product betweeen vectors
        n_dot_v = self.dot(normal, v).clamp(min=0)
        n_dot_l = self.dot(normal, l).clamp(min=0)
        n_dot_h = self.dot(normal, h).clamp(min=0)
        v_dot_h = self.dot(v, h).clamp(min=0)

        # lambert brdf
        f1 = diffuse / np.pi
        f1 = f1 * (1 - specular)

        # cook-torrence brdf
        D = self.GGX(n_dot_h, roughness**2)
        F = self.Fresnel(v_dot_h, specular)
        G = self.Smith(n_dot_v, n_dot_l, roughness**2)
        f2 = D * F * G / (4 * n_dot_v * n_dot_l + self.eps)

        # brdf
        kd = 1; ks = 1
        f = kd * f1 + ks * f2

        # rendering
        img = self.light_pow * f * n_dot_l / dist_l_sq

        # print("[DONE:Microfacet] rendering")
        return img.clamp(0, 1)
