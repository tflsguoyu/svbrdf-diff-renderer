# -*- coding: utf-8 -*-
#
# Copyright (c) 2024, Yu Guo. All rights reserved.

import torch as th

from .capture import Capture
from .materialgan import MaterialGANOptim
from .microfacet import Microfacet
from .svbrdf import SvbrdfIO, SvbrdfOptim


def gen_textures_from_materialgan(json_dir):
    device = th.device("cuda:0" if th.cuda.is_available() else "cpu")
    # device = th.device("cpu")

    svbrdf_obj = SvbrdfIO(json_dir, device)

    optim_obj = MaterialGANOptim(device, None, "ckp/materialgan.pth")
    optim_obj.init_from([])
    textures = optim_obj.latent_to_textures(optim_obj.latent)

    svbrdf_obj.save_textures_th(textures, svbrdf_obj.reference_dir)


def render(json_dir, res):
    device = th.device("cuda:0" if th.cuda.is_available() else "cpu")
    # device = th.device("cpu")

    svbrdf_obj = SvbrdfIO(json_dir, device)
    textures = svbrdf_obj.load_textures_th(svbrdf_obj.reference_dir, res)

    render_obj = Microfacet(res, svbrdf_obj.n_of_imgs, svbrdf_obj.im_size, svbrdf_obj.cl, device)
    rendereds = render_obj.eval(textures)

    svbrdf_obj.save_images_th(rendereds, svbrdf_obj.target_dir)


def gen_targets_from_capture(data_dir):
    input_obj = Capture(data_dir)
    input_obj.eval(size=17.0, depth=0.1)


def optim_perpixel(json_dir, res, lr, epochs, tex_init):
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
        textures = svbrdf_obj.load_textures_th(svbrdf_obj.reference_dir, res)
        optim_obj.init_from_tex(textures)
    else:
        exit()

    optim_obj.optim(epochs, lr, svbrdf_obj)

    svbrdf_obj.save_textures_th(optim_obj.textures.clamp(-1, 1), svbrdf_obj.optimize_dir)
    rendereds = renderer_obj.eval(optim_obj.textures.clamp(-1, 1))
    svbrdf_obj.save_images_th(rendereds, svbrdf_obj.rerender_dir)


def optim_ganlatent(json_dir, res, lr, epochs, tex_init):
    # epochs: list of 3 int. [0] is total epochs, [1] is epochs for latent, [2] is for noise, in each cycle.
    # tex_init: string. [], [PATH_TO_LATENT.pt], or [PATH_TO_LATENT.pt, PATH_TO_NOISE.pt]

    device = th.device("cuda:0" if th.cuda.is_available() else "cpu")
    # device = th.device("cpu")

    svbrdf_obj = SvbrdfIO(json_dir, device)
    targets = svbrdf_obj.load_images_th(svbrdf_obj.target_dir, res)

    renderer_obj = Microfacet(res, svbrdf_obj.n_of_imgs, svbrdf_obj.im_size, svbrdf_obj.cl, device)

    optim_obj = MaterialGANOptim(device, renderer_obj, ckp="ckp/materialgan.pth")
    optim_obj.load_targets(targets)

    optim_obj.init_from(tex_init)

    optim_obj.optim(epochs, lr, svbrdf_obj)

    svbrdf_obj.save_textures_th(optim_obj.textures, svbrdf_obj.optimize_dir)
    rendereds = renderer_obj.eval(optim_obj.textures)
    svbrdf_obj.save_images_th(rendereds, svbrdf_obj.rerender_dir)
