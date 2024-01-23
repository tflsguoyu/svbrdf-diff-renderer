# -*- coding: utf-8 -*-
#
# Copyright (c) 2023, Yu Guo. All rights reserved.

import os
from pathlib import Path
import numpy as np
import torch as th

from src.microfacet import Microfacet
from src.svbrdf import SvbrdfIO


def render(json_dir):

    device = th.device("cuda:0" if th.cuda.is_available() else "cpu")
    # device = th.device("cpu")

    svbrdf_obj = SvbrdfIO(json_dir, device)
    render_obj = Microfacet(256, svbrdf_obj.n_of_imgs, svbrdf_obj.im_size, svbrdf_obj.cl, device)

    textures = svbrdf_obj.load_textures_th(svbrdf_obj.reference_dir)
    rendereds = render_obj.eval(textures)

    svbrdf_obj.save_images_th(rendereds, svbrdf_obj.target_dir)


if __name__ == "__main__":
    json_dir = Path("data/card_blue/target.json")
    render(json_dir)