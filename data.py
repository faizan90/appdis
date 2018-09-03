'''
Created on Aug 8, 2018

@author: Faizan-Uni
'''
from timeit import default_timer
import psutil

import numpy as np
import pandas as pd

from depth_funcs import gen_usph_vecs_norm_dist_mp as gen_usph_vecs


class AppearDisappearData:

    '''Set the data for the AppearDisappearAnalysis class.

    This is a base class.
    '''

    def __init__(
            self, verbose=True, copy_input=False, mutability=False):

        '''
        Parameters
        ----------
        verbose : bool
            Print activity messages if True.
        copy_input : bool
            Make copies of input data if True. This will result in more
            memory consumption.
        mutability : bool
            The writable flag of all input arrays. If False arrays become
            read-only. This applies regardless of the value of copy_input.
        '''

        assert isinstance(verbose, bool)
        assert isinstance(copy_input, bool)
        assert isinstance(mutability, bool)

        self.verbose = verbose
        self._copy_input = copy_input
        self._mtbl_flag = mutability

        self._poss_t_idx_types = ['time', 'range']

        self._data_arr_set_flag = False
        self._time_index_set_flag = False
        self._uvecs_set_flag = False
        self._data_vrfd_flag = False
        return

    def set_data_array(self, data_arr):

        '''Set the time series data array for the appearing and disappearing
        events' analysis.

        Parameters
        ----------
        data_arr : 2D float64 np.ndarray
            Rows represent timesteps while columns represent variables.
            Only finite values are allowed. Number of rows and columns
            should be greater than zero.
        '''

        assert isinstance(data_arr, np.ndarray), 'data_arr not a numpy array!'
        assert data_arr.ndim == 2, 'data_arr can be 2D only!'
        assert data_arr.shape[0] > 0, 'Rows should be greater than zero!'
        assert data_arr.shape[1] > 0, 'Columns should be greater than zero!'
        assert np.all(np.isfinite(data_arr)), 'Invalid values in data_arr!'

        if not self._copy_input:
            assert data_arr.flags.c_contiguous, 'data_arr not c_contiguous!'
            assert data_arr.dtype.type is np.float64, (
                'data_arr dtype is not np.float64!')

        if self._time_index_set_flag:
            assert self._t_idx.shape[0] == data_arr.shape[0], (
                'Unequal lengths of data_arr and time_index!')

        if self._copy_input:
            self._data_arr = np.array(data_arr, dtype=np.float64, order='c')

        else:
            self._data_arr = data_arr

        self._data_arr.flags.writeable = self._mtbl_flag

        self._n_data_pts = data_arr.shape[0]
        self._n_data_dims = data_arr.shape[1]

        if self.verbose:
            print(3 * '\n', 50 * '#', sep='')
            print(f'Data array set with {self._n_data_pts} rows and '
                  f'{self._n_data_dims} columns.')

        self._data_arr_set_flag = True
        return

    def set_time_index(self, time_index):

        '''Set the time series index array for the appearing and disappearing
        events' analysis.

        Parameters
        ----------
        time_index : 1D int64 np.ndarray, pd.DatetimeIndex
            An array representing the time step values for each row
            of the data_arr. It should be increasing in magnitude.
        '''

        if isinstance(time_index, pd.DatetimeIndex):

            time_index_type = 'time'

            _dfs = (time_index.view(np.int64)[1:] -
                    time_index.view(np.int64)[:-1])

        elif isinstance(time_index, np.ndarray):

            time_index_type = 'range'

            _dfs = time_index[1:] - time_index[:-1]

            if not self._copy_input:
                assert time_index.flags.c_contiguous, (
                    'time_index not c-contiguous!')
                assert time_index.dtype.type is np.int64, (
                'time_index should have the dtype of np.int64!')

        else:
            raise NotImplementedError

        assert time_index.ndim == 1
        assert np.all(_dfs > 0)

        if self._data_arr_set_flag:
            assert time_index.shape[0] == self._n_data_pts, (
                'Unequal lengths of data_arr and time_index!')

        if self._copy_input:
            if time_index_type == 'time':
                self._t_idx = pd.DatetimeIndex(time_index)

            elif time_index_type == 'range':
                self._t_idx = np.array(time_index, dtype=np.int64, order='c')
                self._t_idx.flags.writeable = self._mtbl_flag

        else:
            self._t_idx = time_index

        self._t_idx_t = time_index_type

        if self.verbose:
            print(3 * '\n', 50 * '#', sep='')
            print(f'Time index set with a total length of '
                  f'{self._t_idx.shape[0]} and type: {self._t_idx_t}.')

        self._time_index_set_flag = True
        return

    def set_unit_vectors(self, uvecs=None):

        '''Set the unit vectors used for the depth function.

        Parameters
        ----------
        uvecs : 2D float64 np.ndarray
            Each row is a unit vector with columns less than equal to that
            of the data_arr columns. Actual dimensions depend on the user.
        '''

        assert isinstance(uvecs, np.ndarray), 'uvecs not a numpy arrays!'
        assert uvecs.ndim == 2, 'uvecs can be 2D only!'
        assert np.all(np.isfinite(uvecs)), 'Invalid values in uvecs!'
        assert uvecs.shape[0] > 0, 'No unit vectors in uvecs!'

        if not self._copy_input:
            assert uvecs.flags.c_contiguous, 'uvecs not c-contiguous!'
            assert uvecs.dtype.type is np.float64, (
                'uvecs should have the dtype of np.float64!')

        if self._data_arr_set_flag:
            assert uvecs.shape[1] <= self._n_data_dims, (
                'uvecs have more columns than those of data_arr!')

        if self._copy_input:
            self._uvecs = np.array(uvecs, dtype=np.float64, order='c')

        else:
            self._uvecs = uvecs

        self._uvecs.flags.writeable = self._mtbl_flag

        self._n_uvecs = uvecs.shape[0]

        self._ans_dims = uvecs.shape[1]

        if self.verbose:
            print(3 * '\n', 50 * '#', sep='')
            print(f'Unit vectors set with {self._n_uvecs} points and '
                  f'{self._uvecs.shape[0]} dimensions.')

        self._uvecs_set_flag = True
        return

    def generate_and_set_unit_vectors(
            self, n_uvec_dims, n_uvecs, n_cpus='auto'):

        '''Generate and set unit vectors for the depth function.

        Parameters
        ----------
        n_uvec_dims : int
            Dimensions of unit vectors to generate.
        n_uvecs : int
            Number of unit vectors to generate.
        n_cpus : str, int
            Number of threads used to generate the vectors.
            If 'auto' then use maximum available threads - one.
        '''

        assert isinstance(n_uvec_dims, int), 'n_uvec_dims not an integer!'
        assert isinstance(n_uvecs, int), 'n_uvecs not an integer!'

        assert (n_uvec_dims > 0), (
            'Dimensions of unit vectors cannot be less than one!')
        assert (n_uvecs > 0), (
            'Number of unit vectors cannot be less than one!')

        if n_cpus != 'auto':
            assert isinstance(n_cpus, int), 'n_cpus not an integer!'
            assert (n_cpus > 0), (
                'Number of processing threads cannot be less than one!')

        else:
            n_cpus = max(1, psutil.cpu_count() - 1)

        if self._data_arr_set_flag:
            assert n_uvec_dims <= self._n_data_dims, (
                'uvecs have more columns than those of data_arr!')

        if self.verbose:
            print(3 * '\n', 50 * '#', sep='')
            print(f'Generating {n_uvecs} unit vectors with {n_uvec_dims} '
                  f'dimensions using {n_cpus} threads...')

        begt = default_timer()
        uvecs = gen_usph_vecs(n_uvecs, n_uvec_dims, n_cpus)
        tott = default_timer() - begt

        self._uvecs = uvecs
        self._n_uvecs = uvecs.shape[0]
        self._ans_dims = uvecs.shape[1]

        self._uvecs.flags.writeable = self._mtbl_flag

        if self.verbose:
            print(f'Done generating unit vectors in {tott: 0.3f} secs.')

        self._uvecs_set_flag = True
        return

    def verify(self):

        '''Verify that all the inputs are correct.

        NOTE:
        -----
            These are just some additional checks. This function should
            always be called after all the inputs are set and ready.
        '''

        assert self._data_arr_set_flag, 'Call set_data_array first!'
        assert self._time_index_set_flag, 'Call set_time_index first!'
        assert self._uvecs_set_flag, (
            'Call set_unit_vectors or generate_and_set_unit_vectors first!')

        assert self._t_idx.shape[0] == self._n_data_pts, (
            'Unequal lengths of data_arr and time_index!')

        assert self._ans_dims <= self._n_data_dims, (
            'uvecs have more columns than those of data_arr!')

        uvec_mags = (self._uvecs ** 2).sum(axis=1) ** 0.5
        assert np.all(np.isclose(uvec_mags, 1.0)), (
            'unit vectors have magnitudes unequal to one!')

        if self.verbose:
            print(3 * '\n', 50 * '#', sep='')
            print('All data inputs verified to be correct.')

        self._data_vrfd_flag = True
        return

    def get_data_array(self):

        assert self._data_arr_set_flag, 'Call set_data_array first!'
        return self._data_arr

    def get_time_index(self):

        assert self._time_index_set_flag, 'Call set_time_index first!'
        return self._t_idx

    def get_unit_vectors(self):

        assert self._uvecs_set_flag, (
            'Call set_unit_vectors or generate_and_set_unit_vectors first!')
        return self._uvecs

    __verify = verify
