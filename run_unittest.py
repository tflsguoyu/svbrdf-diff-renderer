# -*- coding: utf-8 -*-
#
# Copyright (c) 2023, Yu Guo. All rights reserved.

from pathlib import Path
from src.imageio import img9to1, tex4to1

data_dir = Path("data/card_blue")

img9to1(data_dir / "images/reference/256")
tex4to1(data_dir / "textures/reference/256")
