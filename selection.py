'''
Created on Aug 22, 2018

@author: Faizan-Uni
'''
from timeit import default_timer

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

    def set_optimization_parameters(
            self,
            initial_annealing_temperature,
            temperature_reduction_alpha,
            update_at_every_iteration_no,
            maximum_iterations,
            maximum_without_change_iterations,
            maximum_allowed_correlation):

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
        maximum_allowed_correlation : float
            Maximum correlation that any two vectors can have in the
            correlation matrix in order for it to be accepted as an
            optimized configuration.
            It is an absoulte value between 0 and 1.
        '''

        assert isinstance(initial_annealing_temperature, float)
        assert isinstance(temperature_reduction_alpha, float)
        assert isinstance(update_at_every_iteration_no, int)
        assert isinstance(maximum_iterations, int)
        assert isinstance(maximum_without_change_iterations, int)
        assert isinstance(maximum_allowed_correlation, float)

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

        assert 0 < maximum_allowed_correlation < 1

        self._iat = initial_annealing_temperature
        self._tra = temperature_reduction_alpha
        self._uaein = update_at_every_iteration_no
        self._mis = maximum_iterations
        self._mwocis = maximum_without_change_iterations
        self._mac = maximum_allowed_correlation

        if self.verbose:
            print(3 * '\n', 50 * '#', sep='')
            print(f'Set the following optimzation parameters:')
            print(f'\tInitial annealing temperature: {self._iat}')
            print(f'\tTemperature reduction alpha: {self._tra}')
            print(f'\tUpdate iteration no: {self._uaein}')
            print(f'\tMax. iterations: {self._mis}')
            print(f'\tMax without change iterations: {self._mwocis}')
            print(f'\tMax allowed correlation: {self._mac}')

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

        if self.verbose:
            print(3 * '\n', 50 * '#', sep='')
            print('All optimization inputs verified to be correct.')

        self._opt_vrfd_flag = True
        return

    def generate_vector_indicies_set(self):

        '''Find the indicies of vectors in data array such that the sum of
        correlation among all of them is minimum of all the other possible
        combinations. The number of vectors considered at any given iteration
        is equal to the dimensions of the unit vectors.
        '''

        if self.verbose:
            print(3 * '\n', 50 * '#', sep='')
            print('Finding most uncorrelated vectors...')

        self._bef_opt()

        assert self._opt_vrfd_flag, (
            'Optimization inputs unverified. Call verify first!')

        osel_idxs = np.random.choice(
            self._irng, size=self._ans_dims, replace=False)

        old_corr_arr = self._acorr_arr[osel_idxs][:, osel_idxs].copy()

        old_obj_val = self._get_obj_ftn_val(old_corr_arr)

        i_obj_vals = []
        acc_vals = 0
        acc_rates = []
        min_obj_vals = []

        ci = 0
        cwoci = 0
        i = 0
        ctp = self._iat

        begt = default_timer()

        if self._ans_dims == self._irng.shape[0]:
            # no need to optimize
            i = self._mis

        while (i < self._mis) and (cwoci < self._mwocis):

            if ci > self._uaein:
                ci = 0
                ctp = ctp * self._tra

            old_idx = np.random.choice(
                osel_idxs, size=1, replace=False)[0]

            new_idx = np.random.choice(self._irng, size=1, replace=False)[0]

            ctr = 0
            while (old_idx == new_idx) or (new_idx in osel_idxs):

                new_idx = np.random.choice(
                    self._irng, size=1, replace=False)[0]

                if ctr > 100:
                    raise RuntimeError('Something wrong is!')

                ctr += 1

            nsel_idxs = osel_idxs.copy()

            nsel_idxs[nsel_idxs == old_idx] = new_idx

            new_corr_arr = self._acorr_arr[nsel_idxs][:, nsel_idxs].copy()

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
                osel_idxs = nsel_idxs
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

        tott = default_timer() - begt

        assert np.all(np.isclose(
            old_corr_arr, self._acorr_arr[osel_idxs][:, osel_idxs])), (
                'This should not happen!')

        self._fidxs = np.sort(osel_idxs)
        self._fca = self._acorr_arr[self._fidxs][:, self._fidxs]

        if self.verbose:
            print('\n')
            print('Objective function value:', old_obj_val)
            print('Optimization iterations:', i)
            print('Total optimization iterations:', self._mis)
            print('Final indicies:', self._fidxs)
            print('Final correlation array:')
            print(self._fca)

        assert self._fca.max() <= self._mac, (
            f'Could not find a configuration with minimum '
            f'correlations below {self._mac}. Please increase '
            f'maximum_allowed_correlation or reduce the dimensions '
            f'of the problem.')

        self._siovs = np.array(i_obj_vals)
        self._smovs = np.array(min_obj_vals)
        self._sars = np.array(acc_rates)

        if self.verbose:
            print(f'Done finding most uncorrelated vectors in '
                  f'{tott: 0.3f} secs.')

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

    def _get_obj_ftn_val(self, corrs_arr):

        return corrs_arr.sum() + corrs_arr.max()

    def _bef_opt(self):

        '''Prepare variables required by the optimization.'''

        assert self._opt_vrfd_flag, (
            'Optimization inputs unverified. Call verify first!')

        # don't use this for anything else except this optimization
        self._acorr_arr = np.abs(np.corrcoef(self._data_arr.T))

        # diagonal set to zero
        self._acorr_arr.ravel()[::self._acorr_arr.shape[0] + 1] = 0

        self._acorr_arr.flags.writeable = self._mtbl_flag

        self._irng = np.arange(self._n_data_dims)
        return

    __verify = verify
