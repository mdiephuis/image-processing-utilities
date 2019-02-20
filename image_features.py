# Real domain image filters and operations
import numpy as np
import cv2

from scipy import ndimage, signal
from skimage.filters import threshold_otsu
from scipy.ndimage.filters import median_filter
from sklearn.preprocessing import scale
from scipy.ndimage.filters import convolve
from skimage.feature import local_binary_pattern

from .utilities import to_normed_uint8


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


def gram_matrix(x, normalize=True):
    g = x.dot(x.T)
    if normalize:
        g /= (g.shape[0] * g.shape[1])
    return g


def w_stat(im, mode):
    im = to_normed_uint8(im)
    acc = np.zeros((256, 256))

    if mode == 'w':
        xx = im[:, :-1].ravel()
        yy = im[:, 1:].ravel()
    elif mode == 'h':
        xx = im[:-1, :].ravel()
        yy = im[1:, :].ravel()
    elif mode == 'du':
        xx = im[:-5, :].ravel()
        yy = im[5:, :].ravel()
    elif mode == 'dd':
        xx = im[:, :-5].ravel()
        yy = im[:, 5:].ravel()
    else:
        return None

    for x, y in zip(xx, yy):
        acc[x, y] += 1

    return acc


def im2spin(im):
    im = med_filt(im)
    bw = im > threshold_otsu(im)
    bw = bw.astype(np.int64, copy=False)
    bw[bw == 0] = -1

    return bw


def med_filt(im):
    f = median_filter(im, size=3)
    g = np.abs(im - f)
    g = scale(g, axis=0, with_mean=True, with_std=True, copy=True)
    return g


def binarize(im, method='std'):

    if method == 'std':
        threshold = np.mean(im.ravel()) - 2 * np.std(im.ravel())
    else:
        threshold = threshold_otsu(im)

    bw = im > threshold
    bw = bw.astype(np.float64, copy=False)
    return bw


# Energy map util functions
def gradxy(im_in, blur=False):
    im = im_in.copy()
    if blur:
        im = cv2.GaussianBlur(im, (5, 5), 0)

    gradX = cv2.Sobel(im, ddepth=cv2.CV_64F, dx=1, dy=0, ksize=-1)
    gradY = cv2.Sobel(im, ddepth=cv2.CV_64F, dx=0, dy=1, ksize=-1)
    gradXY = cv2.convertScaleAbs(gradX + gradY)

    return gradXY


def grad_sobel(im_in, blur=False):
    im = im_in.copy()
    du = np.array([[1., 2., 1.], [0., 0., 0.], [-1., -2., -1.]])
    dv = du.T
    energy_map = np.absolute(convolve(im, du)) + np.absolute(convolve(im, dv))
    return energy_map


def grad_lbp(im, radius=3, method='uniform'):
    n_points = 8 * radius
    lbp = local_binary_pattern(im, n_points, radius, method)
    return lbp
