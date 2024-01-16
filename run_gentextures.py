# -*- coding: utf-8 -*-
#
# Copyright (c) 2023, Yu Guo. All rights reserved.

import os
from pathlib import Path
import numpy as np
import torch as th

from src.microfacet import Microfacet
from src.svbrdf import SvbrdfIO, SvbrdfOptim
from src.materialgan import MaterialGANOptim


def optim_materialgan(data_dir, res):

    device = th.device("cuda:0" if th.cuda.is_available() else "cpu")
    # device = th.device("cpu")

    svbrdf_obj = SvbrdfIO(data_dir, device)
    renderer_obj = Microfacet(res, svbrdf_obj.n_of_imgs, svbrdf_obj.im_size, svbrdf_obj.cl, device)
    optim_obj = MaterialGANOptim(device, "tool/materialgan.pth")

    optim_obj.init_latent()
    optim_obj.latent_to_textures()
    svbrdf_obj.save_textures_th(optim_obj.textures, "reference", res)

    rendereds = renderer_obj.eval(optim_obj.textures)
    svbrdf_obj.save_images_th(rendereds, "reference", res)


if __name__ == "__main__":
    data_dir = Path("data/random_textures")
    optim_materialgan(data_dir, 256)