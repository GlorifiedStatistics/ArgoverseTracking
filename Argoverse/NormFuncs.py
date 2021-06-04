"""
Normalization utilities/functions to help normalize the data for better model performance
"""

from utils.Utils import *


def noop(arr, **kwargs):
    """
    Perform no operation, return the same dataset
    """
    return arr, {}


def inv_noop(arr, **kwargs):
    return arr


def linear(arr, mins=None, maxs=None, vmin=0, vmax=1, **kwargs):
    """
    Min-max normalization
    """
    mins = np.min(arr, axis=0) if mins is None else mins
    maxs = np.max(arr, axis=0) if maxs is None else maxs
    ranges = maxs - mins
    ranges[ranges == 0] = 1
    return ((arr - mins) / ranges) * (vmax - vmin) + vmin, {'mins': mins, 'ranges': ranges, 'vmin': vmin, 'vmax': vmax}


def inv_linear(arr, vmin=0, vmax=1, mins=None, ranges=None, **kwargs):
    if mins is None or ranges is None:
        raise ValueError("Need to pass in norm_func kwargs.")
    return ((arr - vmin) / (vmax - vmin)) * ranges + mins


def tanh(arr, **kwargs):
    return np.tanh(arr), {}


def inv_tanh(arr, **kwargs):
    arr[arr >= 1] = 0.999999
    arr[arr <= -1] = -0.999999
    return np.arctanh(arr)


def standardize(arr, means=None, stds=None, **kwargs):
    means = np.mean(arr, axis=0) if means is None else means
    stds = np.std(arr, axis=0) if stds is None else stds
    stds[stds == 0] = 1
    return (arr - means) / stds, {'means': means, 'stds': stds}


def inv_standardize(arr, means=None, stds=None, **kwargs):
    if means is None or stds is None:
        raise ValueError("Need to pass in norm_func kwargs.")
    return arr * stds + means


def get_norm_funcs(norm_func):
    """
    Get the norm_func from the string name, or None (noop)
    """
    if norm_func is None:
        f1, f2 = noop, inv_noop
    else:
        norm_func = norm_func.lower()
        if norm_func in ['noop', 'none', 'no']:
            f1, f2 = noop, inv_noop
        elif norm_func in ['linear', 'lin', 'minmax', 'mm', 'min_max', 'min-max']:
            f1, f2 = linear, inv_linear
        elif norm_func in ['tanh']:
            f1, f2 = tanh, inv_tanh
        elif norm_func in ['standardize']:
            f1, f2 = standardize, inv_standardize
        else:
            raise ValueError("Unknown norm_func: %s" % norm_func)

    def _ret(d, inv=False, **kwargs):
        """
        To normalize on differently dimensioned data
        """
        if len(d.shape) == 1:
            ns = [1, -1]
            return f2(d.reshape(ns), **kwargs).reshape([-1]) if inv else f1(d.reshape(ns), **kwargs).reshape([-1])
        elif len(d.shape) == 2:
            return f2(d, **kwargs) if inv else f1(d, **kwargs)
        else:
            raise ValueError("Norm func data has wrong number of dimensions!")

    return _ret
