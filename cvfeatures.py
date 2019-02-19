# Computer vision features
import numpy as np
import cv2
import mahotas


# bin order
# Needs modification into second feature
def _lbp_seq(seq):
    pivot = len(seq) // 2
    thresh = seq[pivot]
    return [1 * (x >= thresh) for x in seq[:pivot] + seq[pivot + 1:]]


# Big endian
def _encode(seq, P):
    lbp = 0
    for i in range(P):
        lbp += seq[i] * (2**(P - 1) >> i)
    return lbp


def _bit_rotate_right(val, shift):
    # Cyclic bit shift to the right
    return (val >> 1) | ((val & 1) << (shift - 1))


def _rot2(lbp, P):
    rot_chain = np.zeros(P, dtype=np.int32)
    rot_chain[0] = lbp
    for i in range(1, P):
        rot_chain[i] = _bit_rotate_right(rot_chain[i - 1], P)
    lbp = rot_chain[0]
    for i in range(1, P):
        lbp = min(lbp, rot_chain[i])
    return lbp


def lbp_feat(s, P, method):

    # Stop working with sequences?
    lbp = 0
    # signed texture
    signed_seq = _lbp_seq(list(s))

    # Weights for small endian encoding
    weights = 2 ** np.arange(P, dtype=np.int32)

    # Uniform method
    if method == 'U':
        # 1) determine the number of 0-1 changes
        n_change = 0
        for i in range(P - 1):
            n_change += (signed_seq[i] - signed_seq[i + 1]) != 0

        if n_change <= 2:
            for i in range(P):
                lbp += signed_seq[i]
        else:
            lbp = P + 1
    # Default or ROR
    else:
        # Default
        for i in range(P):
            lbp += signed_seq[i] * weights[i]
        # ROR
        if method == 'R':
            lbp = _rot2(lbp, P)

    # Write out
    return lbp


def patch2lbp(patch):
    lbp = np.zeros((patch.shape[0] - 2, patch.shape[1] - 2), dtype=int)
    for i in range(1, patch.shape[0] - 1):
        for j in range(1, patch.shape[1] - 1):
            code = int(0)
            center = patch[i, j]
            code = code | (patch[i - 1, j - 1] >= center) * 1 << 0
            code = code | (patch[i - 1, j] >= center) * 1 << 1
            code = code | (patch[i - 1, j + 1] >= center) * 1 << 2
            code = code | (patch[i, j + 1] >= center) * 1 << 3
            code = code | (patch[i + 1, j + 1] >= center) * 1 << 4
            code = code | (patch[i + 1, j] >= center) * 1 << 5
            code = code | (patch[i + 1, j - 1] >= center) * 1 << 6
            code = code | (patch[i, j - 1] >= center) * 1 << 7
            lbp[i - 1, j - 1] = code
    return lbp


def seq2patch(seq):
    patch = np.zeros((3, 3), dtype=int)
    patch[0, 0] = seq[0]
    patch[0, 1] = seq[1]
    patch[0, 2] = seq[2]
    patch[1, 0] = seq[8]
    patch[1, 1] = seq[4]
    patch[1, 2] = seq[3]
    patch[2, 0] = seq[7]
    patch[2, 1] = seq[6]
    patch[2, 2] = seq[5]
    return patch


def patch2seq(patch):
    seq = [patch[0, 0], patch[0, 1], patch[0, 2], patch[1, 2], patch[1, 1], patch[2, 2], patch[2, 1], patch[2, 0], patch[1, 0]]
    return seq


def get_lbp(patch, P, method):
    seq = patch2seq(patch)
    return lbp_feat(seq, P, method)


# signal wrappers and histogram
def signal2lbp(x, stride=1, method='uniform'):
    f_size = 4
    P = 8
    lbp = []
    for i in range(f_size, x.shape[0] - f_size - 1):
        seq = x[i - f_size:i + f_size + 1]
        lbp_val = lbp_feat(seq, P, method)
        lbp.append(lbp_val)

    return np.asarray(lbp)


def hist(ax, lbp):
    n_bins = int(lbp.max() + 1)
    return ax.hist(lbp.ravel(), normed=True, bins=n_bins, range=(0, n_bins),
                   facecolor='0.5')


def lbp_hist(lbp):
    n_bins = int(lbp.max() + 1)
    hist, _ = np.histogram(lbp, normed=True, bins=n_bins, range=(0, n_bins))
    return hist


def hsv_hist(im, n_bins):
    hsv_im = cv2.cvtColor(im, cv2.COLOR_RGB2HSV)
    hist = cv2.calcHist([hsv_im], [0, 1, 2], None, n_bins, [0, 180, 0, 256, 0, 256])
    # Magic line, normalize and flatten (8, 12, 13)
    hist = cv2.normalize(hist, hist).flatten()
    return hist


def hu_moments(im):
    if len(im.shape) > 2:
        im = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
    feature = cv2.HuMoments(cv2.moments(im)).flatten()
    return feature


def haralick(im):
    gray = im.copy()
    if len(gray.shape) > 2:
        gray = cv2.cvtColor(gray, cv2.COLOR_BGR2GRAY)
    feature = mahotas.features.haralick(gray).mean(axis=0)
    return feature


def cvfeature_to_array(cv_keypoints):
    # Extracts pt_x, pt_y, angle, octave, response, size from N cv2 SIFT keypoints
    # and returns (N, 6) ndarray.
    if not isinstance(cv_keypoints, list):
        return cvfeature_to_array(list(cv_keypoints))

    f = np.zeros((len(cv_keypoints), 6))
    for ind, kp in enumerate(cv_keypoints):
        f[ind, 0] = kp.pt[0]
        f[ind, 1] = kp.pt[1]
        f[ind, 2] = kp.angle
        f[ind, 3] = kp.octave
        f[ind, 4] = kp.response
        f[ind, 5] = kp.size

    return f


def filter_feature(f, filter, n_pts=None):
    fmap = {
        'angle': 2,
        'octave': 3,
        'response': 4,
        'size': 5
    }

    ind = np.argsort(f[:, fmap[filter]])[::-1]

    if n_pts is None:
        return f[ind, :], ind
    else:
        n_elem = min(n_pts, f.shape[0])
        return f[ind[:n_elem], :], ind[:n_elem]
