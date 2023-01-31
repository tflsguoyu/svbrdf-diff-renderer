# -*- coding: utf-8 -*-
#
# Copyright (c) 2023, Yu Guo. All rights reserved.

import os

from src.render import render


DATA_DIR = 'data'

TEXS_DIR = os.path.join(DATA_DIR, 'tex')  # Textures folder
PARS_DIR = os.path.join(DATA_DIR, 'par')  # Parameters folder
RDRS_DIR = os.path.join(DATA_DIR, 'rdr')  # Renderings folder

render(TEXS_DIR, PARS_DIR, RDRS_DIR)
render(TEXS_DIR, PARS_DIR=False, RDRS_DIR)
