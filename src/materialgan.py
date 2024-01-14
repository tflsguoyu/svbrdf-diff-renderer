# -*- coding: utf-8 -*-
#
# Copyright (c) 2024, Yu Guo. All rights reserved.

from src.optimization import Optim

class MaterialGANOptim(Optim):

    def __init__(self, device, loss_type="L2"):
        super().__init__(device, loss_type)


    def optim(self, epochs, lr=0.01):
        self.optimizer = th.optim.Adam(self.textures, lr=lr, betas=(0.9, 0.999))
        self.iteration(epochs)
