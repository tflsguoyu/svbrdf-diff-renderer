# -*- coding: utf-8 -*-
#
# Copyright (c) 2023, Yu Guo. All rights reserved.

# import os
# import numpy as np

# from src.render import render
# from src.optim import optim
# from src.util import tex4to1


# DATA_DIR = 'data'

# IMGS_DIR = os.path.join(DATA_DIR, 'img')  # Reference folder
# TEXS_DIR = os.path.join(DATA_DIR, 'tex')  # Textures folder
# RDRS_DIR = os.path.join(DATA_DIR, 'rdr')  # Renderings folder

# # render(DATA_DIR)
# optim(DATA_DIR)

# tex4to1(os.path.join(TEXS_DIR, 'optim'))

from pathlib import Path
from src.imageio import imread, img9to1, tex4to1

# img9to1(Path("data/card_blue/images/reference"))
tex4to1(Path("data/card_blue/textures/reference"))
