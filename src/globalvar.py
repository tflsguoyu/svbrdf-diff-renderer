import os
import torch as th
from torch.autograd import Variable

def gradient(parameters, device):
    return th.autograd.Variable(parameters.to(device), requires_grad=True)

def randn_noise(res):
    return th.randn(1, 1, res, res, dtype=th.float32)

def init_global_noise(device, init_from="random"):
    global noises
    global noise_idx
    noise_idx = 0

    if init_from == 'random':
        noise_4_1 = randn_noise(4)
        noise_8_1 = randn_noise(8)
        noise_8_2 = randn_noise(8)
        noise_16_1 = randn_noise(16)
        noise_16_2 = randn_noise(16)
        noise_32_1 = randn_noise(32)
        noise_32_2 = randn_noise(32)
        noise_64_1 = randn_noise(64)
        noise_64_2 = randn_noise(64)
        noise_128_1 = randn_noise(128)
        noise_128_2 = randn_noise(128)
        noise_256_1 = randn_noise(256)
        noise_256_2 = randn_noise(256)
    else:
        if os.path.exists(init_from):
            noises_list = th.load(init_from, map_location=device)
            noise_4_1 = noises_list[0]
            noise_8_1 = noises_list[1]
            noise_8_2 = noises_list[2]
            noise_16_1 = noises_list[3]
            noise_16_2 = noises_list[4]
            noise_32_1 = noises_list[5]
            noise_32_2 = noises_list[6]
            noise_64_1 = noises_list[7]
            noise_64_2 = noises_list[8]
            noise_128_1 = noises_list[9]
            noise_128_2 = noises_list[10]
            noise_256_1 = noises_list[11]
            noise_256_2 = noises_list[12]
        else:
            print('Can not find noise vector ', init_from)
            exit()

    noise_4_1 = gradient(noise_4_1, device)
    noise_8_1 = gradient(noise_8_1, device)
    noise_8_2 = gradient(noise_8_2, device)
    noise_16_1 = gradient(noise_16_1, device)
    noise_16_2 = gradient(noise_16_2, device)
    noise_32_1 = gradient(noise_32_1, device)
    noise_32_2 = gradient(noise_32_2, device)
    noise_64_1 = gradient(noise_64_1, device)
    noise_64_2 = gradient(noise_64_2, device)
    noise_128_1 = gradient(noise_128_1, device)
    noise_128_2 = gradient(noise_128_2, device)
    noise_256_1 = gradient(noise_256_1, device)
    noise_256_2 = gradient(noise_256_2, device)

    noises = [noise_4_1,  noise_8_1,  noise_8_2,  noise_16_1, noise_16_2,
              noise_32_1, noise_32_2, noise_64_1, noise_64_2,
              noise_128_1, noise_128_2, noise_256_1, noise_256_2]
