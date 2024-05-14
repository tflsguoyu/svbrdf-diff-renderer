# -*- coding: utf-8 -*-
#
# Copyright (c) 2024, Yu Guo. All rights reserved.

import numpy as np
import torch as th
import matplotlib.pyplot as plt


class Optim:
    def __init__(self, device, renderer_obj):
        self.device = device
        self.eps = 1e-4
        self.loss_l2 = th.nn.MSELoss().to(device)
        self.renderer_obj = renderer_obj

    def gradient(self, parameters):
        if isinstance(parameters, list):
            for idx, parameter in enumerate(parameters):
                parameters[idx] = th.autograd.Variable(parameter, requires_grad=True)
        else:
            parameters = th.autograd.Variable(parameters, requires_grad=True)
        return parameters

    def load_targets(self, targets):
        raise NotImplementedError('Should be implemented in derived class!')

    def compute_image_loss(self, predicts):
        return self.loss_l2(predicts, self.targets)

    def optim(self, epochs, lr, svbrdf_obj):
        raise NotImplementedError('Should be implemented in derived class!')

    def save_loss(self, losses, labels, save_dir, N):
        plt.figure(figsize=(8, 4))
        for i in range(len(losses)):
            plt.plot(np.log1p(losses[i]), label=labels[i])
        plt.xlim(0, N)
        plt.legend()
        plt.title("log(1+loss)")
        plt.savefig(save_dir)
        plt.close()
