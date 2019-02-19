# General utilities
import numpy as np
import skimage.util as sku
import skimage.color as skc
from skimage import data as skdata
import tqdm
import cv2


def imread(fp, gray=False):
    im = sku.img_as_float(skdata.imread(fp))
    if gray is True:
        im = skc.rgb2gray(im)
    return im


def to_normed_uint8(im):
    minv = np.amin(im)
    maxv = np.amax(im)
    im = im - minv
    im = im / (maxv - minv)
    im = im * 255
    return im.astype(np.uint8)


def unpack(a):
    if isinstance(a, np.ndarray) or isinstance(a, list):
        return a[0]
    else:
        return a


def mask_diag(X):
    assert(X.shape[0] == X.shape[1])
    X = X[~np.eye(X.shape[0], dtype=bool)].reshape(X.shape[0], -1)
    return X


def frames_to_images(fp_in, fp_out):
    '''
    Read in avi file at fp_in
    Write extracted frames as jpg in fp_out
    '''
    cap = cv2.VideoCapture(fp_in)
    fps = cap.get(cv2.CAP_PRO8P_FPS)
    length = cap.get(cv2.CAP_PROP_FRAME_COUNT)
    print("FPS: {0}".format(fps))
    count = 0
    tbar = tqdm.tqdm_notebook(total=length)
    while(cap.isOpened()):
        frame_id = cap.get(1)
        ret, frame = cap.read()
        if ret is not True:
            break
        if frame_id % fps == 0:
            tbar.update(int(fps))

        cv2.imwrite(fp_out + "/frame_%d.jpg" % count, frame)
        count += 1


def visualize_spectrum(im):

    eps = np.max(im[:, :]) * 1e-2
    s1 = np.log(im[:, :] + eps) - np.log(eps)

    img = (s1 * 255 / np.max(s1)).astype(np.uint8)
    return cv2.equalizeHist(img)
