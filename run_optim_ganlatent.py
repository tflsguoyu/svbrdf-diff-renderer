# -*- coding: utf-8 -*-
#
# Copyright (c) 2023, Yu Guo. All rights reserved.

import os
from pathlib import Path
import numpy as np
import torch as th

from src.microfacet import Microfacet
from src.svbrdf import SvbrdfIO
from src.materialgan import MaterialGANOptim


def optim_ganlatent(json_dir, res, epochs):

    device = th.device("cuda:0" if th.cuda.is_available() else "cpu")
    # device = th.device("cpu")

    svbrdf_obj = SvbrdfIO(json_dir, device)
    targets = svbrdf_obj.load_images_th(svbrdf_obj.target_dir, res)

    renderer_obj = Microfacet(res, svbrdf_obj.n_of_imgs, svbrdf_obj.im_size, svbrdf_obj.cl, device)

    optim_obj = MaterialGANOptim(device, renderer_obj, ckp="tool/materialgan.pth")
    optim_obj.load_targets(targets)
    optim_obj.init_from_latent()

    optim_obj.optim(epochs)

    svbrdf_obj.save_textures_th(optim_obj.textures, svbrdf_obj.optimize_dir)
    rendereds = renderer_obj.eval(optim_obj.textures)
    svbrdf_obj.save_images_th(rendereds, svbrdf_obj.rerender_dir)


if __name__ == "__main__":
    json_dir = Path("data/card_blue/optim.json")
    # json_dir = Path("data/yellow_box/optim.json")
    optim_ganlatent(json_dir, 256, 1000)