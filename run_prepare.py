# -*- coding: utf-8 -*-
#
# Copyright (c) 2023, Yu Guo. All rights reserved.

import os
from pathlib import Path
from src.capture import Capture

def prepare(data_dir, res):
	input_obj = Capture(data_dir, res)
	input_obj.eval(size=17.4, depth=0)


if __name__ == "__main__":
	data_dir = Path("data/bath_tile")
	prepare(data_dir, 1024)