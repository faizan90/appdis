'''
Created on Aug 8, 2018

@author: Faizan-Uni
'''

import psutil
from pathlib import Path

import numpy as np


class AppearDisappearSettings:

    def __init__(self, verbose=True):

        assert isinstance(verbose, bool)

        self.verbose = verbose

        self._poss_ans_stls = ['raw', 'peel', 'alt_peel']
        self._poss_time_wins = ['month', 'year', 'range']

        self._bs_flag = False
        self._hdf5_flag = True

        self._ans_prms_set_flag = False
        self._out_dir_set_flag = False
        self._in_vrfd_flag = False
        return

    def set_analysis_parameters(
            self,
            window_size,
            time_window_type,
            analysis_style,
            analyze_dims,
            peel_depth=0,
            n_cpus='auto'):

        assert (window_size > 0) and np.isfinite(window_size)

        assert isinstance(time_window_type, str)
        assert time_window_type in self._poss_time_wins

        assert isinstance(analysis_style, str)
        assert analysis_style in self._poss_ans_stls

        assert (analyze_dims > 1) and np.isfinite(analyze_dims)

        assert (peel_depth >= 0) and np.isfinite(peel_depth)

        if n_cpus != 'auto':
            assert (n_cpus > 0) and np.isfinite(n_cpus)

        else:
            n_cpus = max(1, psutil.cpu_count() - 1)

        self._ws = window_size
        self._twt = time_window_type
        self._ans_stl = analysis_style
        self._ans_dims = analyze_dims
        self._pl_dth = peel_depth
        self._n_cpus = n_cpus

        self._ans_prms_set_flag = True
        return

    def set_boot_strap_on_off(self, n_boots=0):

        assert isinstance(n_boots, int)
        assert (n_boots >= 0) and np.isfinite(n_boots)

        self._bs_flag = bool(n_boots)
        self._n_bs = n_boots
        return

    def set_outputs_directory(self, out_dir, exist_ok=True):

        assert isinstance(out_dir, (str, Path))

        out_dir = Path(out_dir).resolve()

        assert out_dir.parents[0].exists()

        out_dir.mkdir(exist_ok=exist_ok)

        self._out_dir = out_dir

        self._out_dir_set_flag = True
        return

    def save_outputs_to_hdf5_on_off(self, on=True):

        assert isinstance(on, bool)

        self._hdf5_flag = on
        return

    def verify(self):

        assert self._ans_prms_set_flag
        assert self._out_dir_set_flag

        if (self._ans_stl == 'peel') or (self._ans_stl == 'alt_peel'):
            assert self._pl_dth > 0

        self._in_vrfd_flag = True
        return

