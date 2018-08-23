'''
Created on Aug 22, 2018

@author: Faizan-Uni
'''
import numpy as np

from .data import AppearDisappearData as ADDA


class AppearDisappearVectorSelection(ADDA):

    def __init__(
            self, verbose=True, copy_input=False, mutability=False):

        ADDA.__init__(self, verbose, copy_input, mutability)

        self._opt_prms_set_flag = False
        self._gened_idxs_flag = False
        self._opt_vrfd_flag = False
        return

    def _bef_opt(self):

        '''Prepare variables required by the optimization.

        This is a base class.
        '''

        assert self._opt_vrfd_flag, 'Optimization inputs unverified!'

        self._acorr_arr = np.abs(np.corrcoef(self._data_arr.T))

        self._acorr_arr.flags.writeable = self._mtbl_flag

        self._irng = np.arange(self._n_data_dims)
        return

    def set_optimization_parameters(
            self,
            initial_annealing_temperature,
            temperature_reduction_alpha,
            update_at_every_iteration_no,
            maximum_iterations,
            maximum_without_change_iterations):

        '''Set optimization parameters for simulated annealing.

        Parameters
        ----------
        initial_annealing_temperature : float
            Is what it says.
        temperature_reduction_alpha : float
            The ratio by which to reduce the annealing temperature after
            update_at_every_iteration_no number of iteration. It has to be
            between 0 and 1 (both exclusive).
        update_at_every_iteration_no : int
            After how many iteration shall the annealing temperature be
            reduced.
        maximum_iterations : int
            Maximum number of iterations in simulated annealing.
        maximum_without_change_iterations : int
            Terminate the optimization if a better combination is not found
            after maximum_without_change_iterations iterations.
        '''

        assert isinstance(initial_annealing_temperature, float)
        assert isinstance(temperature_reduction_alpha, float)
        assert isinstance(update_at_every_iteration_no, int)
        assert isinstance(maximum_iterations, int)
        assert isinstance(maximum_without_change_iterations, int)

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

        self._iat = initial_annealing_temperature
        self._tra = temperature_reduction_alpha
        self._uaein = update_at_every_iteration_no
        self._mis = maximum_iterations
        self._mwocis = maximum_without_change_iterations

        self._opt_prms_set_flag = True
        return

    def verify(self):

        '''Verify that all the inputs are correct.

        NOTE:
        -----
            These are just some additional checks. This function should
            always be called after all the inputs are set and ready.
        '''

        ADDA._AppearDisappearData__verify(self)

        assert self._opt_prms_set_flag, 'Optimization parameters not set!'

        self._opt_vrfd_flag = True
        return

    def generate_vector_indicies_set(self):

        '''Find the indicies of vectors in data array such that the sum of
        correlation among all of them is minimum of all the other possible
        combinations. The number of vectors considered at any given iteration
        is equal to the dimensions of the unit vectors.
        '''

        self._bef_opt()

        assert self._opt_vrfd_flag, 'Optimization inputs unverified!'

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
            old_corr_arr, self._acorr_arr[old_sel_idxs][:, old_sel_idxs])), (
                'This should not happen!')

        self._fidxs = np.sort(old_sel_idxs)
        self._fca = self._acorr_arr[self._fidxs][:, self._fidxs]

        if self.verbose:
            print('\n')
            print('Objective function value:', old_obj_val)
            print('Final indicies:', self._fidxs)
            print('Final correlation array:')
            print(self._fca)

        self._siovs = np.array(i_obj_vals)
        self._smovs = np.array(min_obj_vals)
        self._sars = np.array(acc_rates)

        self._gened_idxs_flag = True
        return

    def get_final_vector_indicies(self):

        assert self._gened_idxs_flag, (
            'Call generate_vector_indicies_set first!')
        return self._fidxs

    def get_final_correlations_array(self):

        assert self._gened_idxs_flag, (
            'Call generate_vector_indicies_set first!')
        return self._fca

    def _get_obj_ftn_val(self, corrs_arr):  #

        return corrs_arr.sum()

    __verify = verify
