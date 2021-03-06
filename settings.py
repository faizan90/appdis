'''
Created on Aug 8, 2018

@author: Faizan-Uni
'''

import psutil
from pathlib import Path


class AppearDisappearSettings:

    '''Set parameters for the AppearDisappearAnalysis class.

    This is a base class.
    '''

    def __init__(self, verbose=True):

        '''
        Parameters
        ----------
        verbose : bool
            Print activity messages if True.
        '''

        assert isinstance(verbose, bool)

        self.verbose = verbose

        self._poss_ans_stls = ['un_peel', 'peel', 'alt_peel']
        self._poss_time_wins = ['month', 'year', 'range']

        self._bs_flag = False
        self._hdf5_flag = True
        self._fh_flag = 0

        self._vdl = 0

        self._mvds = 7  # max. volume dims

        self._ans_prms_set_flag = False
        self._out_dir_set_flag = False
        self._in_vrfd_flag = False
        return

    def set_analysis_parameters(
            self,
            time_window_type,
            window_size,
            analysis_style,
            peel_depth=0,
            n_cpus='auto',
            time_unit_step_size=0):

        '''Set the basic parameters for the appearing-disppearing analysis.

        Parameters
        ----------
        time_window_type : str
            The type of window to use while moving along in time.
            If 'year' then windows_size years events are taken per window.
            If 'month' then windows_size month events are taken per window.
            The last two options allow us to get around the situations
            of leap year and the number of days in each month being unqual.
        window_size : int
            Depending on time_window_type, take window_size steps per
            window.
        analysis_style : str
            Three styles of analysis are allowed.
            If 'un_peel' then appearing-disppearing is taken without any
            modifications. This type of data is called unpeeled.
            If 'peel' then take windows that consists of points with
            depths greater than peel_depth only. These windows are called
            peeled.
            If 'alt_peel' then use peeled windows against unpeeled windows
            and vice versa.
            Setting a successive option computes the values for the all
            the preceeding as well. e.g. 'alt_peel' will compute values for
            'un_peel' and 'peel' cases as well.
        peel_depth : int
            Points with depths greater than peel_depth are removed to create
            the peeled window dataset.
        n_cpus : str, int
            Number of threads used by the depth function.
            If 'auto' then use one less than the maximum available threads.
        time_unit_step_size : int
            The number of points that define number of steps per time unit
            if the time_window_type is 'range'. Its value is analogous to the
            number of days in a year if daily data is used and
            time_window_type is 'year'. For 'month' and 'year'
            time_window_type it is ignored.
        '''

        assert isinstance(window_size, int), 'window_size not an integer!'
        assert isinstance(peel_depth, int), 'peel_depth not an integer!'

        assert window_size > 0, 'window_size should be greater than zero!'
        assert peel_depth >= 0, (
            'peel_depth should be greater than or equal to zero!')

        assert isinstance(time_window_type, str), (
            'time_window_type not a string!')

        assert time_window_type in self._poss_time_wins, (
            f'time_window_type can only be one of {self._poss_time_wins}!')

        assert isinstance(analysis_style, str), (
            'analysis_style not a string!')

        assert analysis_style in self._poss_ans_stls, (
            f'analysis_style can only be one of {self._poss_ans_stls}!')

        if analysis_style == 'un_peel':
            peel_depth = 0

        if n_cpus != 'auto':
            assert isinstance(n_cpus, int), 'n_cpus not an integer!'
            assert n_cpus > 0

        else:
            n_cpus = max(1, psutil.cpu_count() - 1)

        if time_window_type == 'range':
            assert isinstance(time_unit_step_size, int)
            assert time_unit_step_size > 0
            assert window_size >= time_unit_step_size
            self._tuss = time_unit_step_size

        else:
            self._tuss = 0

        self._ws = window_size
        self._twt = time_window_type
        self._ans_stl = analysis_style
        self._pl_dth = peel_depth
        self._n_cpus = n_cpus

        self._pl_flag = (
            (self._ans_stl == 'peel') or (self._ans_stl == 'alt_peel'))

        if self.verbose:
            print(3 * '\n', 50 * '#', sep='')
            print(f'Set the following analysis parameters:')
            print(f'\tWindow size: {self._ws}')
            print(f'\tTime window type: {self._twt}')
            print(f'\tAnalysis style: {self._ans_stl}')
            print(f'\tAnalysis dimensions: {self._ans_dims}')
            print(f'\tPeeling depth: {self._pl_dth}')
            print(f'\tN. cpus: {self._n_cpus}')
            print(f'\tTime unit step size: {self._tuss}')

        self._ans_prms_set_flag = True
        return

    def set_outputs_directory(self, out_dir, exist_ok=True):

        '''
        Set the outputs directory.

        Parameters
        ----------
        out_dir : str, pathlib.Path
            Path to the outputs directory
        exist_ok : bool
            Overwrite if output exists
        '''

        assert isinstance(out_dir, (str, Path)), (
            'out_dir not a string or pathlib.Path object!')

        assert isinstance(exist_ok, bool), 'exist_ok not a boolean object!'

        out_dir = Path(out_dir).resolve()

        assert out_dir.parents[0].exists(), (
            'Parent directory of out_dir does not exist!')

        out_dir.mkdir(exist_ok=exist_ok)

        self._out_dir = out_dir

        if self.verbose:
            print(3 * '\n', 50 * '#', sep='')
            print('Set analysis outputs directory to:', str(out_dir))

        self._out_dir_set_flag = True
        return

    def set_boot_strap_on_off(self, n_boots=0):

        '''Allow for bootstrapping in the analysis.

        Parameters
        ----------
        n_boots : int
            Number of boostrap samples to take.
        '''

        assert isinstance(n_boots, int), 'n_boots not an integer!'
        assert n_boots >= 0, 'n_boots cannot be less than zero!'

        self._bs_flag = bool(n_boots)
        self._n_bs = n_boots

        if self.verbose:
            print(3 * '\n', 50 * '#', sep='')
            print(f'Number of bootstraps: {self._n_bs}.')
        return

    def set_volume_boot_strap_on_off(self, nv_boots=0):

        '''Allow for volume based bootstrapping in the analysis.

        Parameters
        ----------
        nv_boots : int
            Number of boostrap samples to take.
        '''

        assert isinstance(nv_boots, int), 'nv_boots not an integer!'
        assert nv_boots >= 0, 'nv_boots cannot be less than zero!'

        self._vbs_flag = bool(nv_boots)
        self._n_vbs = nv_boots

        if self.verbose:
            print(3 * '\n', 50 * '#', sep='')
            print(f'Number of volume bootstraps: {self._n_vbs}.')
        return

    def save_outputs_to_hdf5_on_off(self, on, flush_flag):

        '''Allow for saving of outputs to an HDF5 file.

        Parameters
        ----------
        on : bool
            If True then save analysis outputs to a file 'app_dis_ds.hdf5'
            in the outputs directory.
        flush_flag : int
            At which step to flush outputs to the hdf5 file.
            If 0 then flush at the end of the analysis.
            If 1 then flush after being done with a reference window.
            If 2 then flush after being done with every test window.
            This comes in handy when doing analysis with relatively
            long time series and dimensions. Broken analyses can be
            resumed from the last flush state.
        '''

        assert isinstance(on, bool), 'on can only be a bool!'
        assert isinstance(flush_flag, int), 'flush_flag not an integer!'
        assert 0 <= flush_flag <= 2, (
            'flush_flag can only be between zero and two!')

        self._hdf5_flag = on
        self._fh_flag = flush_flag

        if self.verbose:
            print(3 * '\n', 50 * '#', sep='')
            print(f'Write to HDF5 flag: {self._hdf5_flag}.')

        return

    def save_volume_data_level(self, level, loo_flag=False):

        '''Not sure about this one.'''

        # level:
        #    0: save no volume data
        #    1: save unpeeled and peeled volume data
        #    2: save bootstrapping volume data as well

        assert isinstance(level, int), 'level not an integer!'
        assert 0 <= level <= 1, 'level can only be between zero and one!'

        assert isinstance(loo_flag, bool)

        if not level:
            assert not loo_flag

        self._vdl = level
        self._loo_flag = loo_flag

        if self.verbose:
            print(3 * '\n', 50 * '#', sep='')
            print(f'Set the following volume data level parameters:')
            print(f'\tVolume data level: {self._vdl}')
            print(f'\tLeave one out flag: {self._loo_flag}')
        return

    def verify(self):

        '''Verify that all the inputs are correct.

        NOTE
        ----
            These are just some additional checks. This function should
            always be called after all the parameters are set and ready.
        '''

        assert self._ans_prms_set_flag, 'Call set_analysis_parameters first!'
        assert self._out_dir_set_flag, 'Call set_outputs_directory first!'

        if (self._ans_stl == 'peel') or (self._ans_stl == 'alt_peel'):
            assert self._pl_dth > 0, (
                f'For analysis_style: {self._ans_stl}, '
                'peel_depth should be greater than zero!')

        if self.verbose:
            print(3 * '\n', 50 * '#', sep='')
            print('All settings parameters verified to be correct.')

        self._in_vrfd_flag = True
        return

    __verify = verify
