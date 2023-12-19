# -*- coding: utf-8 -*-
#
# Copyright (c) 2023, Yu Guo. All rights reserved.

from pathlib import Path
from src.imageio import imread, img9to1, tex4to1
from run_render import render
from run_optim import optim


data_dir = Path("data/card_blue")

img9to1(data_dir / "images/reference/1024")
tex4to1(data_dir / "textures/reference/256")

optim(data_dir, 256, 100)
render(data_dir, "optimized", 256)
