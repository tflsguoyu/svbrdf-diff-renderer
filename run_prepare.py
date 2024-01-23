# -*- coding: utf-8 -*-
#
# Copyright (c) 2023, Yu Guo. All rights reserved.

import os
from pathlib import Path
from src.capture import Capture

def prepare(data_dir):
    input_obj = Capture(data_dir)
    input_obj.eval(size=17.0, depth=0.1)


if __name__ == "__main__":
    # if HEIC format, please convert images to PNG first
    data_dir = Path("data/yellow_box") 
    prepare(data_dir)
