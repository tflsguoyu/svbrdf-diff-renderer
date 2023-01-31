# -*- coding: utf-8 -*-
#
# Copyright (c) 2023, Yu Guo. All rights reserved.

from src.microfacet import Microfacet 

def render_tf():

def render(tex_dir, par_dir, rdr_dir):
    # print(lp, cp, L)
    textures, tex_res = png2tex(fn_tex)
    # tex2png(textures, 'a.png')
    # exit()
    if res > tex_res:
        print("[Warning in render.py::renderTex()]: request resolution is larger than texture resolution")
        exit()
    renderObj = Microfacet(res=tex_res, size=size)
    im = renderObj.eval(textures, lightPos=lp, \
        cameraPos=cp, light=th.from_numpy(L).cuda())
    im = gyApplyGamma(gyTensor2Array(im[0,:].permute(1,2,0)), 1/2.2)
    im = gyArray2PIL(im)
    if res < tex_res:
        im = im.resize((res, res), Image.LANCZOS)
    if fn_im is not None:
        im.save(fn_im)
    return im
