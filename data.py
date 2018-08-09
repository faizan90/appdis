'''
Created on Aug 8, 2018

@author: Faizan-Uni
'''

import psutil

import numpy as np
import pandas as pd

from depth_funcs import gen_usph_vecs_norm_dist_mp as gen_usph_vecs


class AppearDisappearData:

    def __init__(self, verbose=True, copy_input=False):

        assert isinstance(verbose, bool)
        assert isinstance(copy_input, bool)

        self.verbose = verbose
        self.copy_input = copy_input

        self._poss_t_idx_types = ['time', 'range']

        self._data_arr_set_flag = False
        self._time_index_set_flag = False
        self._uvecs_set_flag = False
        self._in_vrfd_flag = False

        self.mutbl_arrs_flag = False
        return

    def set_data_array(self, data_arr):

        assert isinstance(data_arr, np.ndarray)
        assert data_arr.ndim == 2
        assert data_arr.shape[1] > 1
        assert np.all(np.isfinite(data_arr))

        if not self.copy_input:
            assert data_arr.flags.c_contiguous
            assert data_arr.dtype.type is np.float64

        if self._time_index_set_flag:
            assert self._t_idx.shape[0] == data_arr.shape[0]

        if self.copy_input:
            self._data_arr = np.array(data_arr, dtype=np.float64, order='c')

        else:
            self._data_arr = data_arr

        self._data_arr.flags.writeable = self.mutbl_arrs_flag

        self._n_data_pts = data_arr.shape[0]
        self._n_data_dims = data_arr.shape[1]

        self._data_arr_set_flag = True
        return

    def set_time_index(self, time_index, time_index_type='time'):

        assert time_index_type in self._poss_t_idx_types

        if time_index_type == 'time':
            assert isinstance(time_index, pd.DatetimeIndex)

            _dfs = (time_index.view(np.int64)[1:] -
                    time_index.view(np.int64)[:-1])

        elif time_index_type == 'range':
            assert isinstance(time_index, np.ndarray)

            _dfs = time_index[1:] - time_index[:-1]

            if not self.copy_input:
                assert time_index.flags.c_contiguous
                assert time_index.dtype.type is np.int64

        else:
            raise NotImplementedError

        assert time_index.ndim == 1
        assert np.all(_dfs > 0)

        if self._data_arr_set_flag:
            assert time_index.shape[0] == self._n_data_pts

        if self.copy_input:
            if time_index_type == 'time':
                self._t_idx = pd.DatetimeIndex(time_index)

            elif time_index_type == 'range':
                self._t_idx = np.array(time_index, dtype=np.int64, order='c')
                self._t_idx.flags.writeable = self.mutbl_arrs_flag

        else:
            self._t_idx = time_index

        self._t_idx_t = time_index_type

        self._time_index_set_flag = True
        return

    def set_unit_vectors(self, uvecs=None):

        assert isinstance(uvecs, np.ndarray)
        assert uvecs.ndim == 2
        assert np.all(np.isfinite(uvecs))

        if not self.copy_input:
            assert uvecs.flags.c_contiguous
            assert uvecs.dtype.type is np.float64

        if self._data_arr_set_flag:
            assert uvecs.shape[1] <= self._n_data_dims

        if self.copy_input:
            self._uvecs = np.array(uvecs, dtype=np.float64, order='c')

        else:
            self._uvecs = uvecs

        self._uvecs.flags.writeable = self.mutbl_arrs_flag

        self._n_uvecs = uvecs.shape[0]

        self._uvecs_set_flag = True
        return

    def generate_and_set_unit_vectors(
            self, n_uvec_dims, n_uvecs, n_cpus='auto'):

        assert (n_uvec_dims > 1) and np.isfinite(n_uvec_dims)
        assert (n_uvecs > 0) and np.isfinite(n_uvecs)

        if n_cpus != 'auto':
            assert (n_cpus > 0) and np.isfinite(n_cpus)

        else:
            n_cpus = psutil.cpu_count() - 1
            if n_cpus <= 0:
                n_cpus = 1

        if self._data_arr_set_flag:
            assert n_uvec_dims <= self._n_data_dims

        uvecs = gen_usph_vecs(n_uvecs, n_uvec_dims, n_cpus)

        self._uvecs = uvecs
        self._n_uvecs = uvecs.shape[0]

        self._uvecs.flags.writeable = self.mutbl_arrs_flag

        self._uvecs_set_flag = True
        return

    def verify(self):

        assert self._data_arr_set_flag
        assert self._time_index_set_flag
        assert self._uvecs_set_flag

        assert self._t_idx.shape[0] == self._n_data_pts

        assert self._uvecs.shape[1] <= self._n_data_dims

        self._in_vrfd_flag = True
        return
