'''
Created on Aug 13, 2018

@author: Faizan-Uni
'''

import numpy as np
import pandas as pd


def ret_mp_idxs(n_vals, n_cpus):

    idxs = np.linspace(0, n_vals, n_cpus + 1, endpoint=True, dtype=np.int64)

    idxs = np.unique(idxs)

    assert idxs.shape[0]

    if idxs.shape[0] == 1:
        idxs = np.concatenate((np.array([0]), idxs))

    assert (idxs[0] == 0) & (idxs[-1] == n_vals), idxs
    return idxs


def cnvt_ser_to_mult_dims_df(in_ser, res_freq):

    '''Convert a given time series to a multi-column series by taking a
    given number of values as a row in succession.

    Parameters
    ----------
    in_ser : pandas.Series
        A pandas series object that needs to be converted.
    res_freq : int
        The number of steps to take successively from in_ser to make the
        multi-column series.

    Return
    ------
    out_df : pandas.DataFrame
        The converted multi-column series. For example, we have in_ser
        with 10 values and res_freq as 2. Then the resulting out_df will
        have 5 values and two columns. The index will be the starting index
        of the successive res_freq values in in_ser. So in this case,
        index of the first, the third, the fifth and so on till the
        9th value. Also, if index of in_ser is of the type
        pandas.DatetimeIndex then out_df has the same type otherwise
        it is a list of values of whatever datatype in_ser.index's
        values have.
    '''

    assert isinstance(in_ser, pd.Series), (
        'in_ser can only be a pandas Series object!')

    assert isinstance(res_freq, int), 'res_freq can only be an integer!'
    assert res_freq > 1, 'res-freq has to greater than one!'

    assert in_ser.shape[0] > res_freq, (
        'in_ser has less or equal values than res_freq!')

    res_shape = -1, res_freq

    n_res_vals = in_ser.shape[0] - (in_ser.shape[0] % res_freq)
    assert n_res_vals > 0

    res_vals = in_ser.values[:n_res_vals].reshape(res_shape)

    res_idx = []

    for i in range(0, in_ser.shape[0], res_freq):
        res_idx.append(in_ser.index[i])

    if isinstance(in_ser.index, pd.DatetimeIndex):
        res_idx = pd.DatetimeIndex(res_idx)

    return pd.DataFrame(data=res_vals, index=res_idx[:res_vals.shape[0]])
