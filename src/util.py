# -*- coding: utf-8 -*-
#
# Copyright (c) 2023, Yu Guo. All rights reserved.

import json
import os

os.environ["OPENCV_IO_ENABLE_OPENEXR"] = "1"  # To write arrays in .EXR file

import cv2
import numpy as np
import torch as th


def imread(filename, type=None):
    '''Loads different types of image from file.

    :param filename: A string of image file name.
    :param type: The type of the image. See all the cases below.

    :return: A numpy array of an image. 
    '''
    match type:
        case 'EXR':
            im = cv2.imread(filename, flags=cv2.IMREAD_ANYDEPTH | cv2.IMREAD_UNCHANGED)
            im = im[..., : : -1]

        case 'sRGB':
            im = cv2.imread(filename)
            im = im[..., : : -1]
            im = im.astype('float32') / 255            
            im = im ** 2.2

        case 'lRGB':
            im = cv2.imread(filename)
            im = im[..., : : -1]
            im = im.astype('float32') / 255            

        case 'NORMAL':
            im = cv2.imread(filename)
            im = im[..., : : -1]
            im = im.astype('float32') / 255
            im = im * 2 - 1
            height, width, c = im.shape
            total = height * width
            im = im.reshape((total, c))
            im_norm = np.linalg.norm(im, axis=1)
            im /= np.broadcast_to(im_norm, (c, total)).T
            im = im.reshape((height, width, c))

        case _:
            im = cv2.imread(filename)

    return im


def imwrite(im, filename, type=None):
    '''Saves different types of image to file.

    :param im: A numpy array of an image.
    :param filename: A string of image file name.
    :param type: The type of the image. See all the cases below.
    '''
    match type:
        case 'EXR':
            im = im[..., : : -1]
            cv2.imwrite(filename, im)

        case 'sRGB':
            gamma = 2.2
            im = im.clip(0, 1) ** (1 / gamma)
            im = (im * 255).astype('uint8')
            im = im[..., : : -1]
            cv2.imwrite(filename, im)

        case 'NORMAL':
            im = (im.clip(-1, 1) + 1) / 2
            im = (im * 255).astype('uint8')
            im = im[..., : : -1]
            cv2.imwrite(filename, im)

        case _:
            cv2.imwrite(filename, im)


def load_parameters(folder, device):

    f = open(os.path.join(folder, 'par.json'))
    data = json.load(f)
    f.close()

    im_size = data["image_size"]
    index = data["id"]
    camera_pos = np.array(data["camera_pos"], 'float32')
    light_pos = np.array(data["light_pos"], 'float32')
    light_pow = np.array(data["light_pow"], 'float32')

    n_of_imgs = len(index)
    camera_pos = camera_pos[index, :]
    light_pos = light_pos[index, :]

    camera_pos = th.from_numpy(camera_pos).to(device)
    light_pos = th.from_numpy(light_pos).to(device)
    light_pow = th.from_numpy(light_pow).to(device)

    return n_of_imgs, im_size, (camera_pos, light_pos, light_pow)


def load_textures(folder, device):

    f = open(os.path.join(folder, 'par.json'))
    data = json.load(f)
    f.close()

    tex_folder = data["texture_folder"]

    fn_normal = os.path.join(folder, tex_folder, 'nom.png')
    fn_diffuse = os.path.join(folder, tex_folder, 'dif.png')
    fn_specular = os.path.join(folder, tex_folder, 'spe.png')
    fn_roughness = os.path.join(folder, tex_folder, 'rgh.png')

    normal = imread(fn_normal, 'NORMAL')
    diffuse = imread(fn_diffuse, 'sRGB')
    specular = imread(fn_specular, 'sRGB')
    roughness = imread(fn_roughness, 'sRGB')

    assert(normal.shape[0] == normal.shape[1])
    res = normal.shape[0]

    normal = th.from_numpy(normal).permute(2, 0, 1).unsqueeze(0).to(device)
    diffuse = th.from_numpy(diffuse).permute(2, 0, 1).unsqueeze(0).to(device)
    specular = th.from_numpy(specular).permute(2, 0, 1).unsqueeze(0).to(device)
    roughness = th.from_numpy(roughness).permute(2, 0, 1).unsqueeze(0).to(device)

    return res, (normal, diffuse, specular, roughness)


def image9to1(folder):
    for i in range(9):
        im_this = imread(os.path.join(folder, '%02d.png' % i))
        if i == 0:
            h, w = im_this.shape[0], im_this.shape[1]
            im = np.zeros([h * 3, w * 3, 3], 'float32')
        r, c = int(i / 3), int(i % 3)
        im[r * h : (r + 1) * h, c * w : (c + 1) * w, :] = im_this
    imwrite(im, os.path.join(folder, 'all.png'))


def tex4to1(folder):
    normal = imread(os.path.join(folder, 'nom.png'))
    diffuse = imread(os.path.join(folder, 'dif.png'))
    specular = imread(os.path.join(folder, 'spe.png'))
    roughness = imread(os.path.join(folder, 'rgh.png'))

    h, w = normal.shape[0], normal.shape[1]
    tex = np.zeros([h * 2, w * 2, 3], 'float32')
    tex[h * 0 : h * 1, w * 0 : w * 1, :] = normal
    tex[h * 0 : h * 1, w * 1 : w * 2, :] = diffuse
    tex[h * 1 : h * 2, w * 0 : w * 1, :] = specular
    tex[h * 1 : h * 2, w * 1 : w * 2, :] = roughness

    imwrite(tex, os.path.join(folder, 'tex.png'))
