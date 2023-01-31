# -*- coding: utf-8 -*-
#
# Copyright (c) 2023, Yu Guo. All rights reserved.

import os

os.environ["OPENCV_IO_ENABLE_OPENEXR"] = "1"  # To write arrays in .EXR file

import cv2
import numpy as np


def imread(filename, type):
    '''Loads different types of image from file.

    :param filename: A string of image file name.
    :param type: The type of the image. See all the cases below.

    :return: A numpy array of an image. 
    '''
    match type[:3]:
        case 'EXR':
            im = cv2.imread(filename, flags=cv2.IMREAD_ANYDEPTH | cv2.IMREAD_UNCHANGED)
            im = im[..., : : -1]
        
        case 'RGB':
            im = cv2.imread(filename)
            im = im[..., : : -1]
            im = im.astype('float32') / 255            
            # Convert image from sRGB to linear space
            gamma = float(type[3:])
            im = im ** gamma

        case 'AOC':
            im = cv2.imread(filename, flags=cv2.IMREAD_UNCHANGED)
            assert(im.ndim == 2)
            im = im.astype('float32') / 255

        case 'NOM':
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


def imwrite(im, filename, type):
    '''Saves different types of image to file.

    :param im: A numpy array of an image.
    :param filename: A string of image file name.
    :param type: The type of the image. See all the cases below.
    '''
    match type[:3]:
        case 'EXR':
            im = im[..., : : -1]
            cv2.imwrite(filename, im)

        case 'RGB':
            gamma = float(type[3:])
            im = im.clip(0, 1) ** (1 / gamma)
            im = (im * 255).astype('uint8')
            im = im[..., : : -1]
            cv2.imwrite(filename, im)

        case 'AOC':
            assert(im.ndim == 2)
            im = (im.clip(0, 1) * 255).astype('uint8')
            cv2.imwrite(filename, im)      

        case 'NOM':
            im = (im.clip(-1, 1) + 1) / 2
            im = (im * 255).astype('uint8')
            im = im[..., : : -1]
            cv2.imwrite(filename, im)

        case _:
            cv2.imwrite(filename, im)

