'''
Created on Aug 8, 2018

@author: Faizan-Uni
'''

from pathlib import Path

import h5py
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

plt.ioff()


class AppearDisappearPlot:

    def __init__(self, verbose=True):
        assert isinstance(verbose, bool)

        self.verbose = verbose

        self._h5_path_set_flag = False
        self._out_dir_set_flag = False
        self._fig_size_set_flag = False
        self._in_vrfd_flag = False
        return

    def set_hdf5(self, path):

        assert isinstance(path, (str, Path))

        path = Path(path).resolve()
        assert path.exists()
        assert path.is_file()

        self.h5_path = Path(path)

        self._h5_path_set_flag = True
        return

    def set_outputs_directory(self, out_dir, exist_ok=True):

        assert isinstance(out_dir, (str, Path))

        out_dir = Path(out_dir).resolve()

        assert out_dir.parents[0].exists()

        out_dir.mkdir(exist_ok=exist_ok)

        self._out_dir = out_dir

        self._out_dir_set_flag = True
        return

    def set_fig_size(self, fig_size):

        assert isinstance(fig_size, (tuple, list))
        assert len(fig_size) == 2
        assert (fig_size[0] > 0) and np.isfinite(fig_size[0])
        assert (fig_size[1] > 0) and np.isfinite(fig_size[1])

        self._fgs = fig_size

        self._fig_size_set_flag = True
        return

    def verify(self):

        assert self._h5_path_set_flag
        assert self._out_dir_set_flag

        self._in_vrfd_flag = True
        return

    def _bef_plot(self):

        assert self._in_vrfd_flag

        if not self._fig_size_set_flag:
            self._fgs = (5, 5)

        self._h5_hdl = h5py.File(str(self.h5_path), mode='r')

        dg = self._h5_hdl['in_data']
        self._data_arr = dg['data_arr'][...]

        _t_idx = dg['t_idx'][...]
        self._t_idx = pd.to_datetime(_t_idx, unit='s')

        self._t_idx_t = dg.attrs['t_idx']
        self._uvecs = dg['uvecs'][...]
        self._n_data_pts = dg.attrs['n_data_pts']
        self._n_data_dims = dg.attrs['n_data_dims']
        self._n_uvecs = dg.attrs['n_uvecs']

        ds = self._h5_hdl['settings']
#         self._ws = ds.attrs['ws']
#         self._nms = ds.attrs['nms']
        self._twt = ds.attrs['twt']
        self._ans_stl = ds.attrs['ans_stl']
#         self._ans_dims = ds.attrs['ans_dims']
#         self._pl_dth = ds.attrs['pl_dth']
#         self._n_cpus = ds.attrs['n_cpus']
#         self._out_dir = ds.attrs['out_dir']
        self._bs_flag = ds.attrs['bs_flag']
#         self._n_bs = ds.attrs['n_bs']

        ivs = self._h5_hdl['inter_vars']
        self._mwr = ivs['mwr'][...]
        self._mwi = ivs.attrs['mwi']

        rds = self._h5_hdl['app_dis_arrs']

        self._upld = rds['upld'][...]

        if (self._ans_stl == 'peel') or (self._ans_stl == 'alt_peel'):
            self._pld = rds['pld'][...]

            if self._ans_stl == 'alt_peel':
                self._pld_upld = rds['pld_upld'][...]
                self._upld_pld = rds['upld_pld'][...]

        if self._bs_flag:
            bsds = self._h5_hdl['boot_arrs']

            self._upld_bs_ul = bsds['upld_ul'][...]
            self._upld_bs_ll = bsds['upld_ll'][...]
            self._upld_bs_flg = bsds['upld_flg'][...]

            if (self._ans_stl == 'peel') or (self._ans_stl == 'alt_peel'):

                self._pld_bs_ul = bsds['pld_ul'][...]
                self._pld_bs_ll = bsds['pld_ll'][...]
                self._pld_bs_flg = bsds['pld_flg'][...]

                if self._ans_stl == 'alt_peel':
                    self._pld_upld_bs_ul = bsds['pld_upld_ul'][...]
                    self._pld_upld_bs_ll = bsds['pld_upld_ll'][...]
                    self._pld_upld_bs_flg = bsds['pld_upld_flg'][...]

                    self._upld_pld_bs_ul = bsds['upld_pld_ul'][...]
                    self._upld_pld_bs_ll = bsds['upld_pld_ll'][...]
                    self._upld_pld_bs_flg = bsds['upld_pld_flg'][...]

        return

    def plot_app_dis(self):

        self._bef_plot()

        plt.figure(figsize=self._fgs)
        plt.imshow(self._upld.T, origin='lower')

#         if self._twt == 'year':
#             plt.xticks(range(n_win_years), years_rng[:n_win_years], rotation=90)
#             plt.yticks(range(n_win_years), years_rng[:n_win_years])
#
#             plt.xlabel('Reference Year')
#             plt.ylabel('Test Year')

#         plt.title(f'Appearing and disappearing counts (window size: {ws} years)')

        plt.colorbar(label='Percentage', orientation='horizontal', shrink=0.5)

        out_fig_name = ('upld_plot.png')

        plt.savefig(str(self._out_dir / out_fig_name), bbox_inches='tight')

        return
