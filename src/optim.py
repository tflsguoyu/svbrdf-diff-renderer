# -*- coding: utf-8 -*-
#
# Copyright (c) 2023, Yu Guo. All rights reserved.

import os

import numpy as np
import torch as th

from src.microfacet import Microfacet 
from src.util import load_parameters
from src.util import load_textures
from src.util import load_targets
from src.util import save_textures


def optim(root_dir):

    device = th.device('cuda:0' if th.cuda.is_available() else 'cpu')
    # device = th.device('cpu')

    optim_res = 512
    epochs = 100

    n, size, cl = load_parameters(root_dir, device)
    res, textures = load_textures(root_dir, optim_res, device)  # Loads initial texture maps
    target_res, targets = load_targets(root_dir, optim_res, device)

    # Optimization
    for idx, texture in enumerate(textures):
        textures[idx] = th.autograd.Variable(texture, requires_grad=True)

    optimizer = th.optim.Adam(textures, lr=0.01, betas=(0.9, 0.999))
    criterion = th.nn.MSELoss().to(device)

    render_obj = Microfacet(optim_res, n, size, cl, device)

    for epoch in range(epochs):

        rendereds = render_obj.eval(textures)

        loss = criterion(rendereds, targets)

        if epoch % 10 == 0 or epoch == (epochs - 1):
            print('[%d/%d]: loss: ' % (epoch, epochs), loss.item())

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    save_textures(textures, root_dir)
