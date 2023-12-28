# -*- coding: utf-8 -*-
#
# Copyright (c) 2023, Yu Guo. All rights reserved.

import os
from pathlib import Path
import numpy as np
import torch as th

from src.microfacet import Microfacet
from src.svbrdf import SvbrdfIO, SvbrdfOptim


def optim(data_dir, res, epochs):

    device = th.device('cuda:0' if th.cuda.is_available() else 'cpu')
    # device = th.device('cpu')

    svbrdf_obj = SvbrdfIO(data_dir, device)
    renderer_obj = Microfacet(res, svbrdf_obj.n_of_imgs, svbrdf_obj.im_size, svbrdf_obj.cl, device)
    optim_obj = SvbrdfOptim(device)

    textures = svbrdf_obj.load_textures_th("reference", res)
    targets = svbrdf_obj.load_images_th("reference", "1024", res)

    # optim_obj.load_textures_from_tex(textures)
    # optim_obj.load_textures_from_const(res)
    optim_obj.load_textures_from_randn(res)
    optim_obj.load_targets(targets)
    optim_obj.load_renderer(renderer_obj)

    optim_obj.optim(epochs)
    svbrdf_obj.save_textures_th(optim_obj.textures, "optimized", res)

    rendereds = renderer_obj.eval(optim_obj.textures)
    svbrdf_obj.save_images_th(rendereds, "optimized", res)


if __name__ == '__main__':
    data_dir = Path("data/card_blue")
    optim(data_dir, 256, 1000)