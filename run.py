# -*- coding: utf-8 -*-
#
# Copyright (c) 2023, Yu Guo. All rights reserved.

from pathlib import Path
from src.imageio import img9to1, tex4to1
from src.scripts import render, render_envmap
from src.scripts import gen_textures_from_materialgan, gen_targets_from_capture
from src.scripts import optim_perpixel, optim_ganlatent


# Fast tools to combine images
# img9to1($FOLDER_WITH_AT_LEAST_9_IMAGES)
# tex4to1($FOLDER_WITH_4_TEXTURE_MAPS)


# Generate texture maps using MaterialGAN
if False:
    gen_textures_from_materialgan(Path("data/random/generate.json"))


# Render texture maps
if False:
    render(Path("data/random/render.json"), 1024)


# --------------------- Example (data/yellow_box) ----------------------
material_dir = Path("data/yellow_box-17.0-0.1")

# Capture your own data
if False:
    gen_targets_from_capture(material_dir, size=17.0, depth=0.1)


# Optimize texture maps directly on pixels, baseline method
if False:
    optim_perpixel(material_dir / "optim_pixel_1024.json", 1024, 0.01, 1000, tex_init="random")


# Optimize texture maps by MaterialGAN for resolution 256x256
if False:
    # Different initialization
    # ckp = []  # random latent and noise
    # ckp = ["ckp/latent_const_W+_256.pt", "ckp/latent_const_N_256.pt"]  # embeded latent and noise for constant maps (lower roughness)
    ckp = ["ckp/latent_avg_W+_256.pt"]  # average latent and zero noise

    optim_ganlatent(material_dir / "optim_latent_256.json", 256, 0.02, [1000, 10, 10], ckp)


# Optimize texture maps by MaterialGAN for resolution higher than 256x256
# MaterialGAN only support 256x256, so `res` in optim_ganlatent(_, res, _, _, _, _) should always be 256
# For higher resolution, we optimize 256x256 maps by MaterialGAN first, than use the scale 2x output as the initialization of 512x512 maps
if True:
    optim_ganlatent(material_dir / "optim_latent_256.json", 256, 0.02, [1000, 10, 10], "auto")
    optim_perpixel(material_dir / "optim_pixel_256_to_512.json", 512, 0.01, 20, tex_init="textures")
    optim_perpixel(material_dir / "optim_pixel_512_to_1024.json", 1024, 0.01, 20, tex_init="textures")


if False:
    render_envmap(material_dir / "optim_latent/1024", 256)
