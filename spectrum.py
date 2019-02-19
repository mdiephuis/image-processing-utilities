# Fourier based utilities
import numpy as np
import cv2
import functools
from scipy import fftpack

# FFT fingerprinting code
##

def subtract_gauss(im, k_size=(3, 3)):
    im -= cv2.GaussianBlur(im, k_size, 0)
    return im


def ff_rand_crop(im, sz, func_name=None):

    # Random patch coordinates of size sz
    y_rnd = np.random.randint(im.shape[0] - sz)
    x_rnd = np.random.randint(im.shape[1] - sz)

    # Get (sz, sz) patch
    im = im[y_rnd:y_rnd + sz, x_rnd:x_rnd + sz]

    if func_name is not None:
        im = func_name(im)

    F = np.abs(np.fft.fftshift(np.fft.fft2(im)))

    return F


def build_fingerprint(im, sz, n_avg=64, func_name=subtract_gauss):
    return functools.reduce(lambda x, y: x + y,
                            map(lambda x: ff_rand_crop(im, sz, func_name),
                                range(n_avg))) / n_avg


def dct2(im):
    return fftpack.dct(fftpack.dct(im.T, norm='ortho').T, norm='ortho')


def idct2(coef):
    return fftpack.idct(fftpack.idct(coef.T, norm='ortho').T, norm='ortho')


def block_dct2(im):
    dct = np.zeros(im.shape)
    for i in np.r_[:im.shape[0]:8]:
        for j in np.r_[:im.shape[1]:8]:
            dct[i:(i + 8), j:(j + 8)] = dct2(im[i:(i + 8), j:(j + 8)])
    return dct
