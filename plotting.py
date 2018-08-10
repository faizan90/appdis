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

        # all labels must have a leading underscore
        self._data_vars_labs = [
            '_data_arr',
            '_t_idx',
            '_t_idx_t',
            '_uvecs',
            '_n_data_pts',
            '_n_data_dims',
            '_n_uvecs',
            ]

        self._sett_vars_labs = [
            '_ws',
            '_twt',
            '_ans_stl',
            '_ans_dims',
            '_pl_dth',
            '_n_cpus',
            '_out_dir',
            '_bs_flag',
            '_n_bs',
            '_hdf5_flag',
            '_fh_flag',
            ]

        self._inter_vars_labs = [
            '_mwr',
            '_mwi',
            ]

        self._app_dis_vars_labs = [
            '_dn_flg',
            '_upld',
            '_pld',
            '_pld_upld',
            '_upld_pld',
            ]

        self._boot_vars_labs = [
            '_upld_bs_ul',
            '_upld_bs_ll',
            '_upld_bs_flg',
            '_pld_bs_ul',
            '_pld_bs_ll',
            '_pld_bs_flg',
            '_pld_upld_bs_ul',
            '_pld_upld_bs_ll',
            '_pld_upld_bs_flg',
            '_upld_pld_bs_ul',
            '_upld_pld_bs_ll',
            '_upld_pld_bs_flg',
            ]

        # sequence matters
        self.h5_ds_names = [
            'in_data', 'settings', 'inter_vars', 'app_dis_arrs', 'boot_arrs']

        self.var_labs_list = [
            self._data_vars_labs,
            self._sett_vars_labs,
            self._inter_vars_labs,
            self._app_dis_vars_labs,
            self._boot_vars_labs
            ]

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

        self._h5_path = Path(path)

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
        self._n_ticks = n_ticks

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

    def _bef_plot(self):

        assert self._in_vrfd_flag

        if not self._fig_props_set_flag:
            self._fgs = (5, 5)
            self._nticks = 15
            self._cmap = plt.get_cmap('jet')

        self._h5_hdl = h5py.File(str(self._h5_path), mode='r')

        h5_dss_list = []

        for name in self.h5_ds_names:
            if name not in self._h5_hdl:
                continue

            h5_dss_list.append(self._h5_hdl[name])

        assert h5_dss_list

        n_dss = len(h5_dss_list)

        for i in range(n_dss):
            dss = h5_dss_list[i]
            var_labs = self.var_labs_list[i]

            for lab in var_labs:
                if lab in dss:
                    setattr(self, lab, dss[lab][...])

                elif lab in dss.attrs:
                    setattr(self, lab, dss.attrs[lab])

                elif lab in var_labs:
                    pass

                else:
                    raise KeyError(lab)

        self._h5_hdl.close()

        # conversions applied to some variables because hdf5 cant have them
        # in the format that is used here
        if (self._twt == 'month') or (self._twt == 'year'):

            self._t_idx = pd.to_datetime(self._t_idx, unit='s')

        self._out_dir = Path(self._out_dir)

        # add mwi to nvs for the diagnal being always nan
        nvs = (~np.isnan(self._upld)).sum() + self._mwi
        self._upld.ravel()[:nvs][::self._mwi + 1] = 0

        if (self._ans_stl == 'peel') or (self._ans_stl == 'alt_peel'):

            self._pld.ravel()[:nvs][::self._mwi + 1] = 0

            if self._ans_stl == 'alt_peel':
                self._pld_upld.ravel()[:nvs][::self._mwi + 1] = 0
                self._upld_pld.ravel()[:nvs][::self._mwi + 1] = 0

        if self._bs_flag:
            self._upld_bs_ul.ravel()[:nvs][::self._mwi + 1] = 0
            self._upld_bs_ll.ravel()[:nvs][::self._mwi + 1] = 0

            if (self._ans_stl == 'peel') or (self._ans_stl == 'alt_peel'):

                self._pld_bs_ul.ravel()[:nvs][::self._mwi + 1] = 0
                self._pld_bs_ll.ravel()[:nvs][::self._mwi + 1] = 0

                if self._ans_stl == 'alt_peel':

                    self._pld_upld_bs_ul.ravel()[:nvs][::self._mwi + 1] = 0
                    self._pld_upld_bs_ll.ravel()[:nvs][::self._mwi + 1] = 0

                    self._upld_pld_bs_ul.ravel()[:nvs][::self._mwi + 1] = 0
                    self._upld_pld_bs_ll.ravel()[:nvs][::self._mwi + 1] = 0

        self._bef_plot_vars_set = True
        return

    def _plot_app_dis(self, plot_arr, out_fig_name, arr_lab):

        assert self._bef_plot_vars_set

        plt.figure(figsize=self._fgs)
        plt.axes().set_aspect('equal', 'box')

        if self._twt == 'year':
            vals = np.unique(self._t_idx.year)
            labs = vals

            xylab = 'year'

        elif self._twt == 'month':
            vals = np.unique(self._mwr)
            labs = []
            for val in vals:
                year_idx = np.where(self._mwr == val)[0][0]

                year = self._t_idx[year_idx].year
                mth = self._t_idx[year_idx].month

                labs.append(f'{year}-{mth:02d}')

            xylab = 'month'

        else:
            raise NotImplementedError

        x = np.arange(vals[0] - 0.5, vals[-1] + 2.5 - self._ws, 1)

        xcs, ycs = np.meshgrid(x, x)

        _ps1 = plt.pcolormesh(
            xcs,
            ycs,
            plot_arr * 100,
            cmap=self._cmap)

        ttl = f'''
        Appearing and disappearing situations

        {arr_lab}
        Analysis style: {self._ans_stl}
        Window type: {self._twt}
        {self._ans_dims} dimensions analyzed
        {self._n_uvecs:1.0E} unit vectors
        Peeling depth: {self._pl_dth}
        Window size: {self._ws} {xylab}(s)
        Starting, ending {xylab}(s): {labs[0]}, {labs[-1]}
        '''

        n_tick_vals = x.shape[0] - 1
        inc = max(1, int(n_tick_vals // (self._n_ticks * 0.5)))

        ticks = x[::inc][:-1] + 0.5
        tick_labs = labs[::inc][:x.shape[0] - 1].astype(str)

        plt.xticks(ticks, tick_labs, rotation=90)
        plt.yticks(ticks, tick_labs)

        plt.xlabel(f'Reference {xylab}')
        plt.ylabel(f'Test {xylab}')

        plt.title(ttl, fontdict={'ha': 'right'}, loc='right')

        plt.colorbar(
            label='Percentage', orientation='horizontal', shrink=0.5)

        plt.savefig(str(self._out_dir / out_fig_name), bbox_inches='tight')
        plt.close()
        return

    def _plot_bs_case(
            self,
            rats_arr,
            rats_bs_ul_arr,
            rats_bs_ll_arr,
            out_fig_name,
            arr_lab):

        assert self._bef_plot_vars_set

        rats_arr = np.ma.masked_invalid(rats_arr)
        rats_bs_ul_arr = np.ma.masked_invalid(rats_bs_ul_arr)
        rats_bs_ll_arr = np.ma.masked_invalid(rats_bs_ll_arr)

        (rats_arr,
         rats_bs_ul_arr,
         rats_bs_ll_arr) = (rats_arr * 100,
                            rats_bs_ul_arr * 100,
                            rats_bs_ll_arr * 100)

        plt.figure(figsize=self._fgs)
        pc = (9, 9)

        ax_ul = plt.subplot2grid(pc, (0, 0), rowspan=4, colspan=4)
        ax_ll = plt.subplot2grid(pc, (0, 4), rowspan=4, colspan=4)
        ax_ra = plt.subplot2grid(pc, (4, 0), rowspan=4, colspan=4)
        ax_cp = plt.subplot2grid(pc, (4, 4), rowspan=4, colspan=4)
        ax_l1 = plt.subplot2grid(pc, (0, 8), rowspan=4, colspan=1)
        ax_l2 = plt.subplot2grid(pc, (4, 8), rowspan=4, colspan=1)

        axes = np.array([[ax_ul, ax_ll], [ax_ra, ax_cp]])

        comp_arr = np.zeros_like(rats_arr)

        with np.errstate(invalid='ignore'):
            comp_arr[rats_arr > rats_bs_ul_arr] = +1
            comp_arr[rats_arr < rats_bs_ll_arr] = -1

        cvmin = -1
        cvmax = +1

        rvmin = min(rats_arr.min(), rats_bs_ul_arr.min(), rats_bs_ll_arr.min())
        rvmax = max(rats_arr.max(), rats_bs_ul_arr.max(), rats_bs_ll_arr.max())

        if not np.isfinite(rvmin):
            rvmin = 0

        if not np.isfinite(rvmax):
            rvmax = 0

        if self._twt == 'year':
            vals = np.unique(self._t_idx.year)
            labs = vals

            xylab = 'year'

        elif self._twt == 'month':
            vals = np.unique(self._mwr)
            labs = []
            for val in vals:
                year_idx = np.where(self._mwr == val)[0][0]

                year = self._t_idx[year_idx].year
                mth = self._t_idx[year_idx].month

                labs.append(f'{year}-{mth:02d}')

            xylab = 'month'

        else:
            raise NotImplementedError

        x = np.arange(vals[0] - 0.5, vals[-1] + 2.5 - self._ws, 1)

        xcs, ycs = np.meshgrid(x, x)

        _ps1 = ax_ul.pcolormesh(
            xcs,
            ycs,
            rats_bs_ul_arr,
            cmap=self._cmap,
            vmin=rvmin,
            vmax=rvmax)

        ax_ll.pcolormesh(
            xcs,
            ycs,
            rats_bs_ll_arr,
            cmap=self._cmap,
            vmin=rvmin,
            vmax=rvmax)

        _ps2 = ax_cp.pcolormesh(
            xcs,
            ycs,
            comp_arr,
            cmap=self._cmap._resample(3),
            vmin=cvmin,
            vmax=cvmax)

        ax_ra.pcolormesh(
            xcs,
            ycs,
            rats_arr,
            cmap=self._cmap,
            vmin=rvmin,
            vmax=rvmax)

        ax_ul.set_ylabel(f'Test {xylab}')
        ax_ra.set_ylabel(f'Test {xylab}')
        ax_cp.set_xlabel(f'Reference {xylab}')
        ax_ra.set_xlabel(f'Reference {xylab}')

        ax_ul.set_title('Upper limit')
        ax_ll.set_title('Lower limit')
        ax_cp.set_title('Bounds check')
        ax_ra.set_title('Actual')

        ttl = f'''
        Appearing and disappearing situations

        {arr_lab}
        Analysis style: {self._ans_stl}
        Window type: {self._twt}
        {self._ans_dims} dimensions analyzed
        {self._n_uvecs:1.0E} unit vectors
        {self._n_bs} bootstraps
        Peeling depth: {self._pl_dth}
        Window size: {self._ws} {xylab}(s)
        Starting, ending {xylab}(s): {labs[0]}, {labs[-1]}
        '''

        plt.suptitle(ttl, x=0, y=1, ha='left')  # , fontdict={'ha': 'right'}, loc='right'
#
        plt.tight_layout(rect=(0, 0, 0.85, 0.85))  #
#         plt.subplots_adjust()

        n_tick_vals = x.shape[0] - 1
        inc = max(1, int(n_tick_vals // (self._n_ticks * 0.5)))

        ticks = x[::inc][:-1] + 0.5
        tick_labs = labs[::inc][:x.shape[0] - 1].astype(str)

        for i in range(axes.shape[0]):
            for j in range(axes.shape[0]):
                axes[i, j].set_xticks(ticks)
                axes[i, j].set_yticks(ticks)

                if j == 0:
                    axes[i, j].set_yticklabels(tick_labs)

                else:
                    axes[i, j].set_yticklabels([])

                if i == (axes.shape[0] - 1):
                    axes[i, j].set_xticklabels(tick_labs, rotation=90)

                else:
                    axes[i, j].set_xticklabels([])

                axes[i, j].set_aspect('equal', 'box')

        ax_l1.set_axis_off()
        cb1 = plt.colorbar(
            _ps1,
            ax=ax_l1,
            fraction=0.4,
            aspect=15,
            orientation='vertical')
        cb1.set_label('Percentage')

        ax_l2.set_axis_off()
        cb2 = plt.colorbar(
            _ps2,
            ax=ax_l2,
            fraction=0.4,
            aspect=15,
            extend='both',
            orientation='vertical')

        cb2.set_ticks([-1, 0, +1])
        cb2.set_ticklabels(['Below', 'Within', 'Above'])
        cb2.set_label('Bootstrapping limits')

        plt.savefig(str(self._out_dir / out_fig_name), bbox_inches='tight')
        plt.close()
        return

    def _plot_bs(self):

        self._plot_bs_case(
            self._upld,
            self._upld_bs_ul,
            self._upld_bs_ll,
            'bs_upld_plot.png',
            'Both unpeeled (Bootstrap)')

        if (self._ans_stl == 'peel') or (self._ans_stl == 'alt_peel'):

            self._plot_bs_case(
                self._pld,
                self._pld_bs_ul,
                self._pld_bs_ll,
                'bs_pld_plot.png',
                'Both peeled (Bootstrap)')

            if self._ans_stl == 'alt_peel':
                # case 1
                self._plot_bs_case(
                    self._pld_upld,
                    self._pld_upld_bs_ul,
                    self._pld_upld_bs_ll,
                    'bs_pld_upld_plot.png',
                    'Peeled-unpeeled (Bootstrap)')

                # case 2
                self._plot_bs_case(
                    self._upld_pld,
                    self._upld_pld_bs_ul,
                    self._upld_pld_bs_ll,
                    'bs_upld_pld_plot.png',
                    'Unpeeled-peeled (Bootstrap)')
        return

    def plot_app_dis(self):

        if not self._bef_plot_vars_set:
            self._bef_plot()

        self._plot_app_dis(self._upld, 'upld_plot.png', 'Both unpeeled')

        if (self._ans_stl == 'peel') or (self._ans_stl == 'alt_peel'):

            self._plot_app_dis(self._pld, 'pld_plot.png', 'Both peeled')

            if self._ans_stl == 'alt_peel':
                self._plot_app_dis(
                    self._pld_upld,
                    'pld_upld_plot.png',
                    'Peeled-unpeeled')

                self._plot_app_dis(
                    self._upld_pld,
                    'upld_pld_plot.png',
                    'Unpeeled-peeled')

        if self._bs_flag:
            self._plot_bs()
        return
