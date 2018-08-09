'''
Created on Aug 8, 2018

@author: Faizan-Uni
'''

from pathlib import Path

import h5py
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.colors import Colormap
from matplotlib.cm import cmap_d

plt.ioff()


class AppearDisappearPlot:

    def __init__(self, verbose=True):
        assert isinstance(verbose, bool)

        self.verbose = verbose

        self._h5_path_set_flag = False
        self._out_dir_set_flag = False
        self._fig_props_set_flag = False
        self._bs_vars_loaded_flag = False
        self._bef_plot_vars_set = False
        self._upld_pltd_flag = False
        self._pld_pltd_flag = False
        self._alt_pltd_flag = False
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

    def set_fig_props(self, fig_size, n_ticks, cmap):

        assert isinstance(fig_size, (tuple, list))
        assert len(fig_size) == 2
        assert (fig_size[0] > 0) and np.isfinite(fig_size[0])
        assert (fig_size[1] > 0) and np.isfinite(fig_size[1])

        assert isinstance(n_ticks, int)
        assert (n_ticks > 0) and np.isfinite(n_ticks)

        assert isinstance(cmap, (str, Colormap))

        self._fgs = fig_size
        self._nticks = n_ticks

        if isinstance(cmap, str):
            assert cmap in cmap_d
            self._cmap = plt.get_cmap(cmap)

        else:
            self._cmap = cmap

        self._fig_props_set_flag = True
        return

    def verify(self):

        assert self._h5_path_set_flag
        assert self._out_dir_set_flag

        self._in_vrfd_flag = True
        return

    def _load_bs_vars(self):
        bsds = self._h5_hdl['boot_arrs']

        self._upld_bs_ul = bsds['upld_ul'][...]
        self._upld_bs_ll = bsds['upld_ll'][...]
#         self._upld_bs_flg = bsds['upld_flg'][...]

        if (self._ans_stl == 'peel') or (self._ans_stl == 'alt_peel'):

            self._pld_bs_ul = bsds['pld_ul'][...]
            self._pld_bs_ll = bsds['pld_ll'][...]
#             self._pld_bs_flg = bsds['pld_flg'][...]

            if self._ans_stl == 'alt_peel':
                self._pld_upld_bs_ul = bsds['pld_upld_ul'][...]
                self._pld_upld_bs_ll = bsds['pld_upld_ll'][...]
#                 self._pld_upld_bs_flg = bsds['pld_upld_flg'][...]

                self._upld_pld_bs_ul = bsds['upld_pld_ul'][...]
                self._upld_pld_bs_ll = bsds['upld_pld_ll'][...]
#                 self._upld_pld_bs_flg = bsds['upld_pld_flg'][...]

        self._bs_vars_loaded_flag = True
        return

    def _bef_plot(self):

        assert self._in_vrfd_flag

        if not self._fig_props_set_flag:
            self._fgs = (5, 5)
            self._nticks = 15
            self._cmap = plt.get_cmap('jet')

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
        self._ws = ds.attrs['ws']
        self._twt = ds.attrs['twt']
        self._ans_stl = ds.attrs['ans_stl']
        self._ans_dims = ds.attrs['ans_dims']
        self._pl_dth = ds.attrs['pl_dth']
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
            self._load_bs_vars()
            assert self._bs_vars_loaded_flag

        self._bef_plot_vars_set = True
        return

    def _plot_app_dis(self, plot_arr, out_fig_name):

        plt.figure(figsize=self._fgs)
        plt.axes().set_aspect('equal', 'box')

        add_ttl = ''

        if self._twt == 'year':
            vals = np.unique(self._t_idx.year)

            xcs, ycs = np.mgrid[
                slice(vals[0] - 0.5, vals[-1] + 0.5 + 2 - self._ws, 1),
                slice(vals[0] - 0.5, vals[-1] + 0.5 + 2 - self._ws, 1)]

            plt.pcolormesh(xcs, ycs, plot_arr * 100, cmap=self._cmap)

            plt.xlabel('Reference year')
            plt.ylabel('Test year')

            add_ttl = f'''
            Window size: {self._ws} year(s)
            Starting, ending year(s): {vals[0]}, {vals[-1]}'''

        elif self._twt == 'month':
            vals = np.unique(self._mwr)
            labs = []
            for val in vals:
                year_idx = np.where(self._mwr == val)[0][0]

                year = self._t_idx[year_idx].year
                mth = self._t_idx[year_idx].month

                labs.append(f'{year}-{mth:02d}')

            x = np.arange(vals[0] - 0.5, vals[-1] + 2.5 - self._ws, 1)

            xcs, ycs = np.meshgrid(x, x)

            plt.pcolormesh(xcs, ycs, plot_arr * 100, cmap=self._cmap)

            n_x_labs = 15
            n_tick_vals = x.shape[0] - 1
            inc = max(1, n_tick_vals // n_x_labs)

            ticks = x[::inc][:-1] + 0.5
            tick_labs = labs[::inc][:x.shape[0] - 1]

            plt.xticks(ticks, tick_labs)
            plt.yticks(ticks, tick_labs)

            plt.xlabel('Reference month')
            plt.ylabel('Test month')

            add_ttl = f'''
            Window size: {self._ws} month(s)
            Starting, ending month(s): {labs[0]}, {labs[-1]}'''

        else:
            raise NotImplementedError

        ttl = f'''
        Appearing and disappearing situations

        Analysis style: {self._ans_stl}
        Window type: {self._twt}
        {self._ans_dims} dimensions analyzed
        {self._n_uvecs:1.0E} unit vectors
        {add_ttl}
        '''

        plt.xticks(rotation=90)

        plt.title(ttl, fontdict={'ha': 'right'}, loc='right')

        plt.colorbar(label='Percentage', orientation='horizontal', shrink=0.5)

        plt.savefig(str(self._out_dir / out_fig_name), bbox_inches='tight')
        plt.close()
        return

    def plot_unpeeled_app_dis(self):

        if not self._bef_plot_vars_set:
            self._bef_plot()

        if not self._upld_pltd_flag:
            self._plot_app_dis(self._upld, 'upld_plot.png')

        self._upld_pltd_flag = True
        return

    def plot_peeled_app_dis(self):

        if not self._bef_plot_vars_set:
            self._bef_plot()

        assert self._pl_dth
        assert (self._ans_stl == 'peel') or (self._ans_stl == 'alt_peel')

#         self.plot_unpeeled_app_dis()

        if not self._pld_pltd_flag:
            self._plot_app_dis(self._pld, 'pld_plot.png')

        self._pld_pltd_flag = True
        return

    def plot_alternate_peel_app_dis(self):

        if not self._bef_plot_vars_set:
            self._bef_plot()

        assert self._pl_dth
        assert self._ans_stl == 'alt_peel'

#         self.plot_peeled_app_dis()

        if not self._alt_pltd_flag:

            self._plot_app_dis(self._pld_upld, 'plt_upld_plot.png')
            self._plot_app_dis(self._upld_pld, 'upld_pld_plot.png')

        self._alt_pltd_flag = True
        return
