# -*- coding: utf-8 -*-
#
# Copyright (c) 2023, Yu Guo. All rights reserved.

import os
import numpy as np

from src.render import render
from src.util import image9to1
from src.util import tex4to1


DATA_DIR = 'data'

IMGS_DIR = os.path.join(DATA_DIR, 'img')  # Reference folder
TEXS_DIR = os.path.join(DATA_DIR, 'tex')  # Textures folder
PARS_DIR = os.path.join(DATA_DIR, 'par')  # Parameters folder
RDRS_DIR = os.path.join(DATA_DIR, 'rdr')  # Renderings folder

render(DATA_DIR)

# image9to1(RDRS_DIR)
# tex4to1(TEXS_DIR)
