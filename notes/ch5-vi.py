import numpy as np
from scipy import sparse


def invert_nonzero(arr):
    arr_inv = arr.copy()
    nz = np.nonzero(arr)
    arr_inv[nz] = 1 / arr[nz]
    return arr_inv


def xlogx(arr):
    arrlog = arr.copy()
    nz = np.nonzero(arr)
    arrlog[nz] = arr[nz] * np.log2(arr[nz])
    return arrlog


def vi(x, y):
    pxy = sparse.coo_matrix((np.ones(x.size), (x.ravel(), y.ravel())),
                            dtype=float).tocsr()
    pxy.data /= np.sum(pxy.data)
    px = np.array(pxy.sum(axis=1)).ravel()
    py = np.array(pxy.sum(axis=0)).ravel()
    px_inv = sparse.diags([invert_nonzero(px)], [0])
    py_inv = sparse.diags([invert_nonzero(py)], [0])
    hygx = -(px * xlogx(py_inv.dot(pxy)).sum(axis=0)).sum()
    hxgy = -(py * xlogx(pxy.dot(px_inv)).sum(axis=1)).sum()
    return hygx + hxgy

