# cython: nonecheck=False
# cython: boundscheck=False
# cython: wraparound=False
# cython: cdivision=True
# cython: language_level=3
# cython: infer_types=False
# cython: embedsignature=True

ctypedef double DT_D
ctypedef unsigned long DT_UL


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
