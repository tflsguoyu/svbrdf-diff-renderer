# -*- coding: utf-8 -*-
#
# Copyright (c) 2023, Yu Guo. All rights reserved.

import os
from pathlib import Path
import numpy as np
import torch as th

from src.microfacet import Microfacet
from src.svbrdf import SvbrdfIO


def render(data_dir, opt_ref, res):

    device = th.device('cuda:0' if th.cuda.is_available() else 'cpu')
    # device = th.device('cpu')

    svbrdf_obj = SvbrdfIO(data_dir, device)
    render_obj = Microfacet(res, svbrdf_obj.n_of_imgs, svbrdf_obj.im_size, svbrdf_obj.cl, device)

    textures = svbrdf_obj.load_textures_th(opt_ref, res)
    rendereds = render_obj.eval(textures)

    svbrdf_obj.save_images_th(rendereds, opt_ref, res)


if __name__ == '__main__':
    data_dir = Path("data/card_blue")
    render(data_dir, "reference", 256)