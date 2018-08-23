'''
Created on Aug 22, 2018

@author: Faizan-Uni
'''
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt

plt.ioff()


class AppearDisappearVectorSelection:

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
        self.copy_input = copy_input
        self._mtbl_flag = mutability

        self._data_arr_set_flag = False
        self._opt_prms_set_flag = False
        self._gened_idxs_flag = False
        self._in_vrfd_flag = False
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

        if not self.copy_input:
            assert data_arr.flags.c_contiguous, 'data_arr not c_contiguous!'
            assert data_arr.dtype.type is np.float64, (
                'data_arr dtype is not np.float64!')

        if self.copy_input:
            self._data_arr = np.array(data_arr, dtype=np.float64, order='c')

        else:
            self._data_arr = data_arr

        self._data_arr.flags.writeable = self._mtbl_flag

        self._n_data_pts = data_arr.shape[0]
        self._n_data_dims = data_arr.shape[1]

        if self.verbose:
            print(f'Vector selection data array set with {self._n_data_pts} '
                  f'rows and {self._n_data_dims} columns.')

        self._data_arr_set_flag = True
        return

    def _bef_opt(self):

        assert self._in_vrfd_flag

        self._acorr_arr = np.abs(np.corrcoef(self._data_arr.T))

        self._acorr_arr.flags.writeable = self._mtbl_flag

        self._irng = np.arange(self._n_data_dims)

        return

    def set_optimization_parameters(
            self,
            number_of_indicies_to_optimize,
            initial_annealing_temperature,
            temperature_reduction_alpha,
            update_at_every_iteration_no,
            maximum_iterations,
            maximum_without_change_iterations):

        assert isinstance(number_of_indicies_to_optimize, int)
        assert isinstance(initial_annealing_temperature, float)
        assert isinstance(temperature_reduction_alpha, float)
        assert isinstance(update_at_every_iteration_no, int)
        assert isinstance(maximum_iterations, int)
        assert isinstance(maximum_without_change_iterations, int)

        assert number_of_indicies_to_optimize >= 2

        assert initial_annealing_temperature > 0
        assert np.isfinite(initial_annealing_temperature)

        assert temperature_reduction_alpha > 0
        assert temperature_reduction_alpha < 1

        assert update_at_every_iteration_no > 0

        assert maximum_iterations > 0
        assert update_at_every_iteration_no < maximum_iterations

        assert maximum_without_change_iterations > (
            update_at_every_iteration_no)
        assert maximum_without_change_iterations <= maximum_iterations

        self._ans_dims = number_of_indicies_to_optimize
        self._iat = initial_annealing_temperature
        self._tra = temperature_reduction_alpha
        self._uaein = update_at_every_iteration_no
        self._mis = maximum_iterations
        self._mwocis = maximum_without_change_iterations

        self._opt_prms_set_flag = True
        return

    def verify(self):

        assert self._data_arr_set_flag
        assert self._opt_prms_set_flag

        assert self._ans_dims <= self._n_data_dims

        self._in_vrfd_flag = True
        return

    def generate_vector_indicies_set(self):

        self._bef_opt()

        assert self._in_vrfd_flag

        old_sel_idxs = np.random.choice(
            self._irng, size=self._ans_dims, replace=False)

        old_corr_arr = self._acorr_arr[old_sel_idxs][:, old_sel_idxs].copy()

        old_obj_val = self._get_obj_ftn_val(old_corr_arr)

        i_obj_vals = []
        acc_vals = 0
        acc_rates = []
        min_obj_vals = []

        ci = 0
        cwoci = 0
        i = 0
        ctp = self._iat

        while (i < self._mis) and (cwoci < self._mwocis):

            if ci > self._uaein:
                ci = 0
                ctp = ctp * self._tra

            old_idx = np.random.choice(old_sel_idxs, size=1, replace=False)[0]

            new_idx = np.random.choice(self._irng, size=1, replace=False)[0]

            while (old_idx == new_idx) or (new_idx in old_sel_idxs):
                new_idx = np.random.choice(self._irng, size=1, replace=False)[0]

            new_sel_idxs = old_sel_idxs.copy()

            new_sel_idxs[new_sel_idxs == old_idx] = new_idx

            new_corr_arr = self._acorr_arr[new_sel_idxs][:, new_sel_idxs].copy()

            new_obj_val = self._get_obj_ftn_val(new_corr_arr)

            if new_obj_val < old_obj_val:
                sel_cond = True

            else:
                rand_p = np.random.random()
                boltz_p = np.exp((old_obj_val - new_obj_val) / ctp)

                if rand_p < boltz_p:
                    sel_cond = True

                else:
                    sel_cond = False

            if sel_cond:
                old_sel_idxs = new_sel_idxs
                old_corr_arr = new_corr_arr
                old_obj_val = new_obj_val

                cwoci = 0

                acc_vals += 1

            else:
                cwoci += 1

            i_obj_vals.append(new_obj_val)
            min_obj_vals.append(old_obj_val)
            acc_rates.append(acc_vals / (i + 1))

            ci += 1
            i = i + 1

        assert np.all(np.isclose(
            old_corr_arr, self._acorr_arr[old_sel_idxs][:, old_sel_idxs]))

        self._fidxs = np.sort(old_sel_idxs)
        self._fca = self._acorr_arr[self._fidxs][:, self._fidxs]

        if self.verbose:
            print('\n')
            print('Objective function value:', old_obj_val)
            print('Final indicies:', self._fidxs)
            print('Final correlation array:')
            print(self._fca)

#         if True:
#             self._sa_i_obj_vals = np.array(i_obj_vals)
#             self._sa_min_obj_vals = np.array(min_obj_vals)
#             self._sa_acc_rates = np.array(acc_rates)
#
#             _, obj_ax = plt.subplots(figsize=(20, 10))
#             acc_ax = obj_ax.twinx()
#
#             plt.suptitle(
#                 f'Simulated annealing results for uncorrelated '
#                 f'vectors\' selection ({self._ans_dims} dimensions)')
#
#             a1 = acc_ax.plot(
#                 acc_rates,
#                 color='gray',
#                 alpha=0.5,
#                 label='acc_rate')
#             p1 = obj_ax.plot(
#                 i_obj_vals,
#                 color='red',
#                 alpha=0.5,
#                 label='i_obj_val')
#
#             p2 = obj_ax.plot(
#                 min_obj_vals,
#                 color='darkblue',
#                 alpha=0.5,
#                 label='min_obj_val')
#
#             obj_ax.set_xlabel('Iteration No. (-)')
#             obj_ax.set_ylabel('Objective function value (-)')
#             acc_ax.set_ylabel('Acceptance rate (-)')
#
#             obj_ax.grid()
#
#             ps = p1 + p2 + a1
#             lg_labs = [l.get_label() for l in ps]
#
#             obj_ax.legend(ps, lg_labs, framealpha=0.5)
#
#             plt.savefig(
#                 str(self._out_dir / 'sim_anneal.png'),
#                 bbox_inches='tight')
#             plt.close()

        self._gened_idxs_flag = True
        return

    def get_final_vector_indicies(self):

        assert self._gened_idxs_flag
        return self._fidxs

    def get_final_correlations_array(self):

        assert self._gened_idxs_flag
        return self._fca

    def _get_obj_ftn_val(self, corrs_arr):  #

        return corrs_arr.sum()

    __verify = verify
