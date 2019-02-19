# General image metrics
import numpy as np


def hamming(x, y):
    '''
    only works on 0/1 values
    '''
    return np.count_nonzero(x != y)


def kl_divergence_np(p, q):
    p = np.asarray(p)
    q = np.asarray(q)
    filt = np.logical_and(p != 0, q != 0)
    return np.sum(p[filt] * np.log2(p[filt] / q[filt]))


def chi2_distance_np(p, q, eps=1e-10):
    p = np.asarray(p)
    q = np.asarray(q)
    return 0.5 * np.sum(((p - q) ** 2) / (p + q + eps))


def l2(p, q):
    p = np.asarray(p)
    q = np.asarray(q)
    return np.linalg.norm(p - q)


def pairwise_row_distance(X, fn_metric):
    n_dims = X.shape[0]
    dist_matrix = np.zeros((n_dims, n_dims))
    for i in range(n_dims):
        for j in range(n_dims):
            dist_matrix[i, j] = fn_metric(X[i], X[j])
    return dist_matrix


def dwt(a, b, d=lambda x, y: np.abs(x - y)):
    # cost matrix
    n_rows, n_cols = len(a), len(a)
    cost = np.zeros((n_rows, n_cols))

    # init first row and col
    cost[0, 0] = d(a[0], b[0])
    for i in np.arange(1, n_rows):
        cost[i, 0] = cost[i - 1, 0] + d(a[i], b[0])

    for j in np.arange(1, n_cols):
        cost[0, j] = cost[0, j - 1] + d(a[0], b[j])

    for i in np.arange(1, n_rows):
        for j in np.arange(1, n_cols):
            min_cost = min(cost[i - 1, j], cost[i, j - 1], cost[i - 1, j - 1])
            cost[i, j] += min_cost + d(a[i], b[j])