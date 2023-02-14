# -*- coding: utf-8 -*-
#
# Copyright (c) 2023, Yu Guo. All rights reserved.

import os

import numpy as np
import torch as th

from src.microfacet import Microfacet 
from src.util import image9to1
from src.util import imwrite
from src.util import load_parameters
from src.util import load_textures

def render(root_dir):

    device = th.device('cuda:0' if th.cuda.is_available() else 'cpu')
    # device = th.device('cpu')

    n, size, cl = load_parameters(root_dir, device)
    res, textures = load_textures(root_dir, -1, device)

    render_obj = Microfacet(res, n, size, cl, device)
    rendereds = render_obj.eval(textures)
    
    for i in range(n):
        rendered_this = rendereds[i, :, :, :].permute(1, 2, 0).detach().cpu().numpy()
        fn_rendered = os.path.join(root_dir, 'rdr/%02d.png' % i)
        imwrite(rendered_this, fn_rendered, 'sRGB')

    image9to1(os.path.join(root_dir, 'rdr'))
