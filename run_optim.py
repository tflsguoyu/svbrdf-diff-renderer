# -*- coding: utf-8 -*-
#
# Copyright (c) 2023, Yu Guo. All rights reserved.

import os
from pathlib import Path
import tqdm
import numpy as np
import torch as th

from src.microfacet import Microfacet
from src.util import SvbrdfIO


def optim(data_dir, res, epochs):

    device = th.device('cuda:0' if th.cuda.is_available() else 'cpu')
    # device = th.device('cpu')

    svbrdf_obj = SvbrdfIO(data_dir, device)
    n = svbrdf_obj.n_of_imgs
    size = svbrdf_obj.im_size
    cl = svbrdf_obj.load_parameters_th()

    # load initial textures
    textures = svbrdf_obj.load_textures_th("reference", res)

    # load target images
    targets = svbrdf_obj.load_images_th("reference", "1024", res)

    # initial rendering
    render_obj = Microfacet(res, n, size, cl, device)

    # Optimization
    for idx, texture in enumerate(textures):
        textures[idx] = th.autograd.Variable(texture, requires_grad=True)

    optimizer = th.optim.Adam(textures, lr=0.01, betas=(0.9, 0.999))
    criterion = th.nn.MSELoss().to(device)

    for epoch in tqdm.trange(epochs):
        # compute renderings
        rendereds = render_obj.eval(textures)

        # compute loss
        loss = criterion(rendereds, targets)

        # optimize
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    svbrdf_obj.save_textures_th(textures, "optimized", res)


if __name__ == '__main__':
    data_dir = Path("data/card_blue")
    optim(data_dir, 256, 100)