# -*- coding: utf-8 -*-
#
# Copyright (c) 2023, Yu Guo. All rights reserved.

import os
from pathlib import Path
import numpy as np
import torch as th

from src.util import SvbrdfIO
from src.microfacet import Microfacet
from src.imageio import imwrite, img9to1


def render(data_dir, opt_ref, res):

    device = th.device('cuda:0' if th.cuda.is_available() else 'cpu')
    # device = th.device('cpu')

    svbrdf_obj = SvbrdfIO(data_dir, device)

    n = svbrdf_obj.n_of_imgs
    size = svbrdf_obj.im_size
    cl = svbrdf_obj.load_parameters_th()
    textures = svbrdf_obj.load_textures_th(opt_ref, res)

    render_obj = Microfacet(res, n, size, cl, device)
    rendereds = render_obj.eval(textures)
    
    for i in range(n):
        rendered_this = rendereds[i, :, :, :].permute(1, 2, 0).detach().cpu().numpy()
        fn_rendered = os.path.join(data_dir, 'images/reference/256/%02d.png' % i)
        imwrite(rendered_this, fn_rendered, 'srgb')

    img9to1(data_dir / 'images/reference/256')


if __name__ == '__main__':
    data_dir = Path("data/card_blue")
    render(data_dir, "reference", 256)