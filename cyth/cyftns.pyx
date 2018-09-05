# cython: nonecheck=False
# cython: boundscheck=False
# cython: wraparound=False
# cython: cdivision=True
# cython: language_level=3
# cython: infer_types=False
# cython: embedsignature=True

import numpy as np
cimport numpy as np

ctypedef double DT_D
ctypedef unsigned long DT_UL

cdef extern from 'math.h' nogil:
    cdef DT_D NAN


cdef DT_D get_mean(DT_D[:] in_arr) nogil:

    cdef:
        Py_ssize_t i, n
        DT_D _sum = 0

    n = in_arr.shape[0]

    for i in xrange(n):
        _sum += in_arr[i]

    return _sum / n


cdef DT_D get_variance(DT_D in_arr_mean, DT_D[:] in_arr) nogil:

    cdef:
        Py_ssize_t i, n
        DT_D _sum = 0

    n = in_arr.shape[0]

    for i in xrange(n):
        _sum += (in_arr[i] - in_arr_mean)**2

    return _sum / (n)


cdef DT_D get_covar(
        DT_D in_arr_1_mean,
        DT_D in_arr_2_mean,
        DT_D[:] in_arr_1,
        DT_D[:] in_arr_2) nogil:

    cdef:
        Py_ssize_t i, n
        DT_D _sum = 0

    n = in_arr_1.shape[0]

    for i in xrange(n):
        _sum += (in_arr_1[i] - in_arr_1_mean) * (in_arr_2[i] - in_arr_2_mean)

    return _sum / n


cdef inline DT_D get_correl(
        DT_D in_arr_1_std_dev,
        DT_D in_arr_2_std_dev,
        DT_D arrs_covar) nogil:

    return arrs_covar / (in_arr_1_std_dev * in_arr_2_std_dev)


cpdef DT_D get_corrcoeff(DT_D[:] act_arr, DT_D[:] sim_arr):

    cdef:
        DT_D act_mean, sim_mean, act_std_dev
        DT_D sim_std_dev, covar, correl

    act_mean = get_mean(act_arr)
    sim_mean = get_mean(sim_arr)

    act_std_dev = get_variance(act_mean, act_arr)**0.5
    sim_std_dev = get_variance(sim_mean, sim_arr)**0.5

    covar = get_covar(act_mean, sim_mean, act_arr, sim_arr)
    correl = get_correl(act_std_dev, sim_std_dev, covar)

    return correl


cpdef dict get_asymms_sample(DT_D[:] u, DT_D[:] v):

    cdef:
        Py_ssize_t i, n_vals
        DT_D asymm_1, asymm_2

    n_vals = u.shape[0]

    asymm_1 = 0.0
    asymm_2 = 0.0

    for i in xrange(n_vals):
        asymm_1 += (u[i] + v[i] - 1)**3
        asymm_2 += (u[i] - v[i])**3

    asymm_1 = asymm_1 / n_vals
    asymm_2 = asymm_2 / n_vals

    return {'asymm_1':asymm_1, 'asymm_2':asymm_2}


cpdef np.ndarray get_2d_rel_hist(
    DT_D[:] x_probs, DT_D[:] y_probs, DT_UL cop_bins):

    '''get the bivariate empirical copula'''

    cdef:
        Py_ssize_t i, j
        Py_ssize_t i_row, j_col

        DT_UL tot_pts = x_probs.shape[0], tot_sum

        DT_D u1, u2

        DT_D[:, ::1] hist_arr = (
            np.zeros((cop_bins, cop_bins), dtype=np.float64))

    tot_sum = 0
    for i in range(tot_pts):
        u1 = x_probs[i]
        u2 = y_probs[i]

        i_row = <Py_ssize_t> (u1 * cop_bins)
        j_col = <Py_ssize_t> (u2 * cop_bins)

        hist_arr[i_row, j_col] += 1
        tot_sum += 1

    assert tot_pts == tot_sum, 'Error!'

    for i in range(cop_bins):
        for j in range(cop_bins):
            hist_arr[i, j] = hist_arr[i, j] / tot_pts

    return np.asarray(hist_arr)


cpdef void copy_arr(
        const DT_D[:, ::1] in_arr, 
              DT_D[:, ::1] out_arr,
        const DT_UL[::1] true_idxs,
        const DT_UL axis,
        ) except +:

    '''A function to copy a subset of values from a big array to a smaller one
    based on where ever true_idxs is non-zero. 
    '''

    cdef:
        Py_ssize_t i, j, ctr

        DT_UL cols = in_arr.shape[1]
        DT_UL in_rows = in_arr.shape[0]
        DT_UL out_rows = out_arr.shape[0]

    assert 0 <= axis <= 1

    if axis == 0:
        assert in_rows == true_idxs.shape[0]
        assert cols == out_arr.shape[1]

        ctr = 0
        for i in range(in_rows):
            if not true_idxs[i]:
                continue

            for j in range(cols):
                out_arr[ctr, j] = in_arr[i, j]

            ctr += 1

        for i in range(ctr, out_rows):
            for j in range(cols):
                out_arr[i, j] = NAN

    elif axis == 1:
        assert cols == true_idxs.shape[0]
        assert in_rows == out_rows

        for i in range(in_rows):
            ctr = 0
            for j in range(cols):
                if not true_idxs[j]:
                    continue

                out_arr[i, ctr] = in_arr[i, j]

                ctr = ctr + 1

            for j in range(ctr, cols):
                out_arr[i, j] = NAN
    return