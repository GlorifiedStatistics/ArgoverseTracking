"""
Normalization utilities/functions to help normalize the data for better model performance
"""

from utils.Utils import *


def noop(arr, _ss, **kwargs):
    """
    Perform no operation, return the same dataset
    """
    return arr


def inv_noop(arr, _ss, **kwargs):
    return arr


def linear(arr, _ss, vmin=0, vmax=1, **kwargs):
    """
    Min-max normalization
    """
    return ((arr - _ss['min']) / _ss['range']) * (vmax - vmin) + vmin


def inv_linear(arr, _ss, vmin=0, vmax=1, **kwargs):
    return ((arr - vmin) / (vmax - vmin)) * _ss['range'] + _ss['min']


def tanh(arr, _ss, **kwargs):
    return np.tanh(arr)


def inv_tanh(arr, _ss, **kwargs):
    return np.arctanh(arr)


_M = 1000000000


def std_step(arr, _ss, std_devs=2, vmin=0, vmax=1, std_min=0.25, std_max=0.75, stretch_stds=True, **kwargs):
    """
    Take all data within std_devs standard deviations from the mean, and min-max normalize it to be in the range
    [std_min, std_max]. Then, min-max normalize all data below std_devs standard deviations from the mean to be in
    the range [vmin, std_min], and do the same for data above to be in the range [std_max, v_max].

    This will hopefully have the effect of increasing the fidelity of our models for smaller values (as the std_dev
    of position_step values is < 0.5), while still allowing our model to predict larger values with less fidelity
    as they occur far less often

    :param stretch_stds: Only used if the cutoff value for std_dev deviations is closer to the mean than std_min/std_max

        if False, and (data_mean + std_devs * data_std) < std_max, or
        (data_mean - std_devs * data_std) < std_min, then the cutoff will be increased/decreased to std_max and
        std_min respectively (IE: all data falling within the range [std_min, std_max] will not be changed).

        If True, then the data within std_dev deviations will be stretched to fit in the range [std_min, std_max]
    """
    ret = arr.copy()

    for i in range(2):
        if _ss['std'][i] == 0:
            continue
        smin = min(_M if stretch_stds else std_min, _ss['mean'][i] - std_devs * _ss['std'][i])
        smax = max(-_M if stretch_stds else std_max, _ss['mean'][i] + std_devs * _ss['std'][i])
        xmin = min(smin - _ss['std'][i], _ss['min'][i])
        xmax = max(smax + _ss['std'][i], _ss['max'][i])
        mins = np.argwhere(arr[:, i] < smin).reshape([-1])
        maxs = np.argwhere(arr[:, i] > smax).reshape([-1])
        stds = np.argwhere(np.logical_and(arr[:, i] >= smin, arr[:, i] <= smax)).reshape([-1])

        ret[:, i][mins] = (ret[mins][:, i] - xmin) / (smin - xmin) * (std_min - vmin) + vmin
        ret[:, i][maxs] = (ret[maxs][:, i] - smax) / (xmax - smax) * (vmax - std_max) + std_max
        ret[:, i][stds] = (ret[stds][:, i] - smin) / (smax - smin) * (std_max - std_min) + std_min

    return ret


def inv_std_step(arr, _ss, std_devs=2, vmin=0, vmax=1, std_min=0.25, std_max=0.75, stretch_stds=True, **kwargs):
    ret = arr.copy()

    for i in range(2):
        smin = min(_M if stretch_stds else std_min, _ss['mean'][i] - std_devs * _ss['std'][i])
        smax = max(-_M if stretch_stds else std_max, _ss['mean'][i] + std_devs * _ss['std'][i])
        xmin = min(smin - _ss['std'][i], _ss['min'][i])
        xmax = max(smax + _ss['std'][i], _ss['max'][i])
        mins = np.argwhere(arr[:, i] < std_min).reshape([-1])
        maxs = np.argwhere(arr[:, i] > std_max).reshape([-1])
        stds = np.argwhere(np.logical_and(arr[:, i] >= std_min, arr[:, i] <= std_max)).reshape([-1])

        ret[:, i][mins] = ((ret[mins][:, i] - vmin) / (std_min - vmin)) * (smin - xmin) + xmin
        ret[:, i][maxs] = ((ret[maxs][:, i] - std_max) / (vmax - std_max)) * (xmax - smax) + smax
        ret[:, i][stds] = ((ret[stds][:, i] - std_min) / (std_max - std_min)) * (smax - smin) + smin

    return ret


def standardize(arr, _ss, **kwargs):
    return (arr - _ss['mean']) / _ss['std']


def inv_standardize(arr, _ss, **kwargs):
    return arr * _ss['std'] + _ss['mean']


def std_linear(arr, _ss, vmin=0, vmax=1, **kwargs):
    return linear(standardize(arr, _ss), _ss, vmin=vmin, vmax=vmax, **kwargs)


def inv_std_linear(arr, _ss, vmin=0, vmax=1, **kwargs):
    return inv_standardize(inv_linear(arr, _ss, vmin=vmin, vmax=vmax), _ss, **kwargs)


def _fix_dimensions(func):
    """
    Makes it so we can pass in multiple different types of data (different dimensions) and the normalization
    functions will still work as intended.

    Input should only be either 1-D with any size, or 2-D with shape[1] == 2.

    If data is 1-D, then the size is checked. If the size is <= 60, then it is assumed to be a prediction from our
        model. In this case, it is converted into a [-1, 2] array before normalizing, and data['p_step'] is passed
        to the norm_func if data_type == 'step', or data['raw'] if data_type == 'raw'. The array is then converted
        back to a [-1] shape array before returning

        If the size is > 60, then it is assumed to be a dataset row, and it will be broken apart and normalized
        with multiple different stats before returning again as a single-dimensional array. The first (19 * 60 * 2)
        elements are assumed to be the car positions layed out first by time step, then by car, then by (x,y) values,
        and is converted to a [-1, 2] shape array to be normalized with data['p_step'] if data_type == 'step', and
        data['p_raw'] if data_type == 'raw'. The next 19 * 60 * 2 values (from [19*60*2: 19*60*4]) are treated as the
        velocity values. It is assumed there are at least 19 * 560 * 4 values in the array. If there are more, then
        it is assumed that there are 2 * MAX_LANES * 2 more values in the array for the lane_pos/lane_step and
        lane_norm values that are then normalized in the same way.

    If the data is 2-D, then it is assumed the data is a list of 1-D datapoints that need to be normalized, and the
        1-D normalization above is done for each row in the array

    If the data is any other dimension, then an error is raised
    """

    def _ret(arr, stats, data_type='step', **kwargs):
        if data_type not in ['step', 'raw']:
            raise ValueError("Norm func data_type must be either 'step' or 'raw'"
                             )
        if len(arr.shape) > 2:
            raise ValueError("Norm func data must be either 1 or 2 dimensional.")

        if len(arr.shape) == 2:
            return np.array([_ret(row, stats, data_type, **kwargs) for row in arr])

        if len(arr) <= 60:
            return func(arr.reshape([-1, 2]), stats['p_' + data_type], **kwargs).reshape([-1])

        ret = np.empty(arr.shape)

        t, f = 19 * 60 * 2, 19 * 60 * 4
        s = [19, 60, 2]
        if data_type == 'step':
            p = arr[:t].copy().reshape(s)
            p[0, :, :] = func(p[0, :, :], stats['p_off'], **kwargs)
            p[1:, :, :] = func(p[1:, :, :], stats['p_step'], **kwargs)

            v = arr[t:f].copy().reshape(s)
            v[0, :, :] = func(v[0, :, :], stats['p_off'], **kwargs)
            v[1:, :, :] = func(v[1:, :, :], stats['p_step'], **kwargs)

            ret[:t] = p.reshape([-1])
            ret[t:f] = v.reshape([-1])

        else:
            ret[:t] = func(arr[:t], stats['p_raw'], **kwargs)
            ret[t:f] = func(arr[t:f], stats['v_raw'], **kwargs)

        ls = f + 2 * MAX_LANES
        le = ls + 2 * MAX_LANES
        if len(arr) == le or len(arr) == le + 19 * 60:
            ret[f:ls] = func(arr[f:ls].reshape([-1, 2]), stats['lane_' + data_type], **kwargs).reshape([-1])
            ret[ls:le] = func(arr[ls:le].reshape([-1, 2]), stats['lane_norm'], **kwargs).reshape([-1])

        _s1 = len(arr) == le + 19 * 60
        _s2 = len(arr) == f + 19 * 60
        if _s1 or _s2:
            _idx = le if _s1 else f
            s = arr[_idx:_idx+19*60].copy().reshape([19, 60])
            stats['speed_start']['std'][1] = 1
            stats['speed_step']['std'][1] = 1
            stats['speed']['std'][1] = 1
            s[0, :] = func(np.hstack((s[0:1, :].T, np.zeros([60, 1]))), stats['speed_start'], **kwargs)[:, 0].reshape([-1])
            s[1:, :] = func(np.hstack((s[1:, :].reshape([-1, 1]), np.zeros([18*60, 1]))),
                            stats['speed' if data_type == 'raw' else 'speed_step'], **kwargs)[:, 0].reshape([18, 60])
            _idx = le if _s1 else f
            ret[_idx:_idx + 19 * 60] = s.reshape([-1])

        return ret

    return _ret


def get_norm_funcs(norm_func):
    """
    Get the norm_func from the string name, or None (noop)
    """
    if norm_func is None:
        return noop, inv_noop

    norm_func = norm_func.lower()
    if norm_func in ['noop', 'none', 'no']:
        return noop, inv_noop
    elif norm_func in ['linear', 'lin', 'minmax', 'mm', 'min_max', 'min-max']:
        f1, f2 = linear, inv_linear
    elif norm_func in ['tanh']:
        f1, f2 = tanh, inv_tanh
    elif norm_func in ['std', 'std_step']:
        f1, f2 = std_step, inv_std_step
    elif norm_func in ['standardize']:
        f1, f2 = standardize, inv_standardize
    elif norm_func in ['std_linear']:
        f1, f2 = std_linear, inv_std_linear
    else:
        raise ValueError("Unknown norm_func: %s" % norm_func)

    return _fix_dimensions(f1), _fix_dimensions(f2)
