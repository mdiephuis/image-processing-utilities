# Real domain image filters and operations
import numpy as np
import cv2

from scipy import ndimage, signal
import scipy.ndimage.filters as filters
import scipy.special as special
import scipy.optimize as optimize
from itertools import chain
import PIL

import skimage.color as skc
import skimage.util as sku
from skimage import data

from scipy.ndimage.filters import laplace

import functools


def local_uniform_variation(im, size):
    mu = ndimage.uniform_filter(im, (size, size))
    mu_square = ndimage.uniform_filter(im**2, (size, size))
    return mu_square - (mu ** 2)


def laplacian(im):
    return cv2.Laplacian(im, cv2.CV_64F)


def var_of_lap(im):
    edge_im = cv2.Laplacian(im, cv2.CV_64F)
    var_edge = np.var(edge_im)
    sum_edge = np.sum(edge_im)
    return sum_edge, var_edge


def subtract_gaussian(im, kernel_size=(3, 3)):
    im_out = im.copy()
    im_out -= cv2.GaussianBlur(im, kernel_size, 0)
    return im_out


def norm_kernel(kernel):
    return kernel / np.sum(kernel)


def gauss_kernel_2d(n, sigma):
    # Build mesh indices and zero center
    Y, X = np.indices((n, n)) - int(n // 2)
    kernel = 1. / (2 * np.pi * sigma ** 2) * np.exp(-(X ** 2 + Y ** 2) / (2 * sigma ** 2))
    return norm_kernel(kernel)


def local_mean(im, kernel):
    return signal.convolve2d(im, kernel, 'same')


def local_standard_deviation(im, kernel):
    # E[X]^2
    mu_square = signal.convolve2d(im ** 2, kernel, 'same')
    # E[X]
    mu = signal.convolve2d(im, kernel, 'same')
    # E[X]^2 - [EX]^2
    return np.sqrt(np.abs(mu_square - mu ** 2))


def local_maximum(im, peak_threshold=0.8, roi=15):
    image = im.copy()
    size = 2 * roi + 1
    im_max = ndimage.maximum_filter(image, size=size, mode='constant')
    mask = (image == im_max)
    image *= mask

    image[:roi] = 0
    image[-roi:] = 0
    image[:, :roi] = 0
    image[:, -roi:] = 0

    im_t = (image > peak_threshold * image.max()) * 1
    f = np.transpose(im_t.nonzero())
    return f, im_t

# var local var
# MRF methods here