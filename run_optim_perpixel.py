# -*- coding: utf-8 -*-
#
# Copyright (c) 2023, Yu Guo. All rights reserved.

import os
from pathlib import Path
import numpy as np
import torch as th

from src.microfacet import Microfacet
from src.svbrdf import SvbrdfIO, SvbrdfOptim


def optim_perpixel(json_dir, res, epochs, tex_init):

    device = th.device("cuda:0" if th.cuda.is_available() else "cpu")
    # device = th.device("cpu")

    svbrdf_obj = SvbrdfIO(json_dir, device)
    targets = svbrdf_obj.load_images_th(svbrdf_obj.target_dir, res)

    renderer_obj = Microfacet(res, svbrdf_obj.n_of_imgs, svbrdf_obj.im_size, svbrdf_obj.cl, device)

    optim_obj = SvbrdfOptim(device, renderer_obj)
    optim_obj.load_targets(targets)

    if tex_init == "random":
        optim_obj.init_from_randn()
    elif tex_init == "const":
        optim_obj.init_from_const()
    elif tex_init == "textures":
        textures = svbrdf_obj.load_textures_th(svbrdf_obj.reference_dir)
        optim_obj.init_from_tex(textures)
    else:
        exit()

    optim_obj.optim(epochs, 0.01, svbrdf_obj)

    svbrdf_obj.save_textures_th(optim_obj.textures, svbrdf_obj.optimize_dir)
    rendereds = renderer_obj.eval(optim_obj.textures)
    svbrdf_obj.save_images_th(rendereds, svbrdf_obj.rerender_dir)


if __name__ == "__main__":
    json_dir = Path("data/card_blue/optim_pixel.json")
    # json_dir = Path("data/yellow_box/optim.json")
    optim_perpixel(json_dir, 256, 1000, tex_init="const")
