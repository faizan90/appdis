'''
Created on Aug 8, 2018

@author: Faizan-Uni
'''
import psutil
from functools import partial
from pathlib import Path
from multiprocessing import Pool

import h5py
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.colors import Colormap
from matplotlib.cm import cmap_d
from scipy.spatial import ConvexHull
from scipy.stats import rankdata
from matplotlib.ticker import MaxNLocator

from depth_funcs import depth_ftn_mp as dftn

from .misc import ret_mp_idxs
from .analysis import AppearDisappearAnalysis
from .settings import AppearDisappearSettings
from .cyth import get_corrcoeff, get_asymms_sample

plt.ioff()


class AppearDisappearPlot:

    def __init__(self, verbose=True):
        assert isinstance(verbose, bool)

        self.verbose = verbose

        adan = AppearDisappearAnalysis()

        self.h5_ds_names = adan.h5_ds_names

        self.var_labs_list = adan.var_labs_list

        adso = AppearDisappearSettings()
        self._poss_ans_stls = adso._poss_ans_stls

        self.dont_read_vars = (
            '_out_dir',  #  it is important to ignore this
            '_dn_flg',
            '_upld_bs_flg',
            '_pld_bs_flg',
            '_pld_upld_bs_flg',
            '_upld_pld_bs_flg')

        self._mp_pool = None
        self._n_cpus = None

        self.loo_flag = False

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

    def set_fig_props(self, n_ticks, cmap):

        assert isinstance(n_ticks, int)
        assert (n_ticks > 0) and np.isfinite(n_ticks)

        assert isinstance(cmap, (str, Colormap))

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

        self._bef_plot()

        self._in_vrfd_flag = True
        return

    def set_n_cpus(self, n_cpus='auto'):

        # must call after verify to take effect

        assert self._in_vrfd_flag

        if n_cpus != 'auto':
            assert (n_cpus > 0) and np.isfinite(n_cpus)

        else:
            n_cpus = max(1, psutil.cpu_count() - 1)

        self._n_cpus = n_cpus
        return

    def _bef_plot(self):

        if not self._fig_props_set_flag:
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

                if lab in self.dont_read_vars:
                    continue

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

        # add mwi to nvs for the diagonal being always nan
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

        plt.figure(figsize=(13, 13))
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

        xcs, ycs = np.meshgrid(x, x, indexing='ij')

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

        plt.figure(figsize=(17, 17))
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

        rvmin = min(
            rats_arr.min(), rats_bs_ul_arr.min(), rats_bs_ll_arr.min())
        rvmax = max(
            rats_arr.max(), rats_bs_ul_arr.max(), rats_bs_ll_arr.max())

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

        xcs, ycs = np.meshgrid(x, x, indexing='ij')

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

        plt.suptitle(ttl, x=0, y=1, ha='left')
#
        plt.tight_layout(rect=(0, 0, 0.85, 0.85))  #

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

    def plot_volumes(self, loo_flag=False):

        assert isinstance(loo_flag, bool)

        self.loo_flag = loo_flag

        assert self._in_vrfd_flag

        self._plot_vols()
        return

    @staticmethod
    def _get_vols(step_idxs, args):

        mp_cond, lab_cond, dims, loo_flag = args[:4]

        labs = []
        vols = []
        loo_vols = []
        n_chull_cts = []
        chull_idxs = []

        if mp_cond:
            path, dts_path, data_path = args[4:]

            h5_hdl = h5py.File(path, driver='core', mode='r')

            dts_arr = h5_hdl[dts_path][...]
            data_arr = h5_hdl[data_path][...]

            h5_hdl.close()

        else:
            dts_arr, data_arr = args[4:]

        for i in step_idxs:
            ct = dts_arr['cts'][i]

            dts = dts_arr['dts'][i, :ct]
            idxs = dts_arr['idx'][i, :ct]

            lab_int = dts_arr['lab'][i]

            if lab_cond == 1:
                lab = lab_int

            elif lab_cond == 2:
                lab = f'{lab_int}'[:4] + '-' + f'{lab_int}'[4:]

            elif lab_cond == 3:
                lab = lab_int

            labs.append(lab)

            bd_idxs = idxs[dts == 1]

            chull_idxs.append(bd_idxs)

            hull_pts = data_arr[bd_idxs, :dims]
            n_chull_pts = bd_idxs.shape[0]

            vols.append(ConvexHull(hull_pts).volume)
            n_chull_cts.append(n_chull_pts)

            if loo_flag:
                # remove a pt and cmpt volume
                loo_idxs = np.ones(n_chull_pts, dtype=bool)
                for j in range(n_chull_pts):
                    loo_idxs[j] = False

                    loo_vols.append(
                        [i, ConvexHull(hull_pts[loo_idxs]).volume])

                    loo_idxs[j] = True

        loo_vols = np.array(loo_vols)
        vols = np.array(vols)
        labs = np.array(labs)
        n_chull_cts = np.array(n_chull_cts)
        chull_idxs = np.unique(np.array(np.concatenate(chull_idxs)))
        return (labs, vols, loo_vols, n_chull_cts, chull_idxs)

    def _prep_for_vols(self, *args):

        labs = []
        vols = []
        loo_vols = []
        n_chull_cts = []
        chull_idxs = []

        if self._twt == 'year':
            lab_cond = 1

        elif self._twt == 'month':
            lab_cond = 2

        elif self._twt == 'range':
            lab_cond = 3

        idxs_rng = np.arange(self._mwi)

        if self._mp_pool is not None:
            mp_cond = True

            mp_idxs = ret_mp_idxs(self._mwi, self._n_cpus)

            part_ftn = partial(
                AppearDisappearPlot._get_vols,
                args=(
                    mp_cond,
                    lab_cond,
                    self._ans_dims,
                    self.loo_flag,
                    *args))

            mwi_gen = (
                idxs_rng[mp_idxs[i]:mp_idxs[i + 1]]

                for i in range(self._n_cpus))

            # use of map is necessary to keep order
            ress = self._mp_pool.map(part_ftn, mwi_gen)

            for res in ress:
                labs.append(res[0])
                vols.append(res[1])
                loo_vols.append(res[2])
                n_chull_cts.append(res[3])
                chull_idxs.append(res[4])

            labs = np.concatenate(labs)
            vols = np.concatenate(vols)
            loo_vols = np.concatenate(loo_vols)
            n_chull_cts = np.concatenate(n_chull_cts)
            chull_idxs = np.concatenate(chull_idxs)

            chull_idxs = np.unique(chull_idxs)

        else:
            mp_cond = False

            dts_arr, = args

            (labs,
             vols,
             loo_vols,
             n_chull_cts,
             chull_idxs) = AppearDisappearPlot._get_vols(
                idxs_rng,
                (mp_cond,
                 lab_cond,
                 self._ans_dims,
                 self.loo_flag,
                 dts_arr,
                 self._data_arr))

        return (labs, vols, loo_vols, n_chull_cts, chull_idxs)

    def _plot_vols(self):

        assert self._bef_plot_vars_set

        assert self._vdl

        if (self._n_cpus > 1) and (self._ans_dims >= 4):

            self._mp_pool = Pool(self._n_cpus)

        if self._mp_pool is not None:
            uvols_res = self._prep_for_vols(
                self._h5_path, '/dts_vars/_rdts', 'in_data/_data_arr')

        else:
            uvols_res = self._prep_for_vols(self._rdts)

        ulabs, uvols, uloo_vols, un_chull_cts, uchull_idxs = uvols_res

        plt_xs = np.arange(len(ulabs))

        vols_fig = plt.figure(figsize=(20, 7))
        npts_fig = plt.figure(figsize=(20, 7))
        bd_pt_fig = plt.figure(figsize=(40, 4))

        plt.figure(vols_fig.number)

        if self.loo_flag:
            plt.scatter(
                uloo_vols[:, 0],
                uloo_vols[:, 1],
                marker='o',
                alpha=0.2,
                label='unpeeled leave-one',
                color='C0',
                zorder=9)

        plt.plot(
            plt_xs,
            uvols,
            marker='o',
            alpha=0.6,
            label='unpeeled',
            color='C0',
            zorder=10)

        plt.figure(npts_fig.number)
        plt.plot(
            plt_xs,
            un_chull_cts,
            marker='o',
            alpha=0.6,
            label='unpeeled',
            color='C0',
            zorder=10)

        plt.figure(bd_pt_fig.number)
        ubd_pt_arr = np.zeros(self._n_data_pts, dtype=np.int16)
        ubd_pt_arr[uchull_idxs] = 1
        plt.plot(
            self._t_idx,
            ubd_pt_arr,
            alpha=0.6,
            label='unpeeled',
            color='C0',
            zorder=10,
            linewidth=1)

        if (self._ans_stl == 'peel') or (self._ans_stl == 'alt_peel'):

            if self._mp_pool is not None:
                pvols_res = self._prep_for_vols(
                    self._h5_path, '/dts_vars/_rpdts', 'in_data/_data_arr')

            else:
                pvols_res = self._prep_for_vols(self._rpdts)

            _, pvols, ploo_vols, pn_chull_cts, pchull_idxs = pvols_res

            plt.figure(vols_fig.number)

            if self.loo_flag:
                plt.scatter(
                    ploo_vols[:, 0],
                    ploo_vols[:, 1],
                    marker='o',
                    alpha=0.2,
                    label='peeled leave-one',
                    color='C1',
                    zorder=5)

            plt.plot(
                plt_xs,
                pvols,
                marker='o',
                alpha=0.6,
                label='peeled',
                color='C1',
                zorder=6)

            plt.figure(npts_fig.number)
            plt.plot(
                plt_xs,
                pn_chull_cts,
                marker='o',
                alpha=0.6,
                label='peeled',
                color='C1',
                zorder=6)

            plt.figure(bd_pt_fig.number)
            pbd_pt_arr = np.zeros(self._n_data_pts, dtype=np.int16)
            pbd_pt_arr[pchull_idxs] = 1
            plt.plot(
                self._t_idx,
                pbd_pt_arr,
                alpha=0.6,
                label='peeled',
                color='C1',
                zorder=6,
                linewidth=1)

        if self._mp_pool is not None:
            self._mp_pool.close()
            self._mp_pool.join()
            self._mp_pool = None

        ttl = f'''
        %s

        Analysis style: {self._ans_stl}
        Window type: {self._twt}
        {self._ans_dims} dimensions analyzed
        {self._n_uvecs:1.0E} unit vectors
        Peeling depth: {self._pl_dth}
        Window size: {self._ws} {self._twt}(s)
        Starting, ending {self._twt}(s): {ulabs[0]}, {ulabs[-1]}
        '''

        # chull volumes
        plt.figure(vols_fig.number)
        ttl_lab = 'Moving window convex hull volumes'
        plt.title(ttl % ttl_lab, fontdict={'ha': 'right'}, loc='right')

        plt.xlabel(f'Window ({self._twt}(s))')
        plt.ylabel(f'{self._ans_dims}D Volume')

        n_tick_vals = self._mwi
        inc = max(1, int(n_tick_vals // (self._n_ticks * 0.5)))

        ticks = plt_xs[::inc]
        tick_labs = ulabs[::inc][:plt_xs.shape[0]]

        plt.xticks(ticks, tick_labs, rotation=90)

        plt.grid()
        plt.legend()

        out_fig_name = 'chull_volumes.png'

        plt.savefig(str(self._out_dir / out_fig_name), bbox_inches='tight')
        plt.close()

        # chull point count
        plt.figure(npts_fig.number)
        ttl_lab = 'Moving window convex hull point count'

        plt.title(ttl % ttl_lab, fontdict={'ha': 'right'}, loc='right')

        plt.xlabel(f'Window ({self._twt}(s))')
        plt.ylabel(f'N C-hull points (-)')

        n_tick_vals = self._mwi
        inc = max(1, int(n_tick_vals // (self._n_ticks * 0.5)))

        ticks = plt_xs[::inc]
        tick_labs = ulabs[::inc][:plt_xs.shape[0]]

        plt.xticks(ticks, tick_labs, rotation=90)

        npts_fig.gca().yaxis.set_major_locator(MaxNLocator(integer=True))

        plt.grid()
        plt.legend()

        out_fig_name = 'chull_point_counts.png'

        plt.savefig(str(self._out_dir / out_fig_name), bbox_inches='tight')
        plt.close()

        # time series of bd points
        plt.figure(bd_pt_fig.number)
        ttl_lab = 'Boundary points time series'

        plt.title(ttl % ttl_lab, fontdict={'ha': 'right'}, loc='right')

        plt.xlabel('Time (steps)')

        plt.xticks(rotation=90)

        plt.yticks([0, 1], ['Not boundary', 'Boundary'])

        npts_fig.gca().yaxis.set_major_locator(
            MaxNLocator(self._n_ticks, integer=True))

        plt.grid()
        plt.legend(framealpha=0.5).set_zorder(11)

        out_fig_name = 'boundary_points_time_series.png'

        plt.savefig(str(self._out_dir / out_fig_name), bbox_inches='tight')
        plt.close()
        return

    def _get_win_bd_idxs(self, dts_arr):

        chull_idxs = []
        idxs_rng = np.arange(self._mwi)

        for i in idxs_rng:
            ct = dts_arr['cts'][i]

            dts = dts_arr['dts'][i, :ct]
            idxs = dts_arr['idx'][i, :ct]

            bd_idxs = idxs[dts == 1]

            chull_idxs.append(bd_idxs)

        _ = np.unique(np.array(np.concatenate(chull_idxs)))

        chull_idxs = np.zeros(self._n_data_pts, dtype=bool)
        chull_idxs[_] = True
        chull_time_idxs = self._t_idx[chull_idxs]

        return (chull_idxs, chull_time_idxs)

    def _get_full_bd_idxs(self, style):

        cd_arr = self._data_arr[:, :self._ans_dims].copy('c')

        refr_refr_dts = dftn(cd_arr, cd_arr, self._uvecs, self._n_cpus)

        upld_chull_idxs = refr_refr_dts <= self._pl_dth
        upld_chull_time_idxs = self._t_idx[upld_chull_idxs]

        pld_chull_idxs = None
        pld_chull_time_idxs = None

        if (style == 'peel') or (style == 'alt_peel'):
            refr_pld = cd_arr[~upld_chull_idxs].copy('c')

            refr_refr_pld_dts = dftn(
                refr_pld, refr_pld, self._uvecs, self._n_cpus)

            pld_chull_idxs = refr_refr_pld_dts <= self._pl_dth
            pld_chull_time_idxs = (
                self._t_idx[~upld_chull_idxs][pld_chull_idxs])

        return (
            upld_chull_idxs,
            upld_chull_time_idxs,
            pld_chull_idxs,
            pld_chull_time_idxs)

    def get_boundary_point_idxs(self, style, data_type):

        assert self._in_vrfd_flag

        assert isinstance(style, str)
        assert style in self._poss_ans_stls

        poss_data_types = ['window', 'full']

        assert isinstance(data_type, str)
        assert data_type in poss_data_types

        if data_type == 'window':
            if style == 'raw':
                dts_arr = self._rdts

            elif (style == 'peel') or (style == 'alt_peel'):

                assert (
                    (self._ans_stl == 'peel') or
                    (self._ans_stl == 'alt_peel'))

                dts_arr = self._rpdts

            else:
                raise NotImplementedError

            res = self._get_win_bd_idxs(dts_arr)

        elif data_type == 'full':
            res = self._get_full_bd_idxs(style)

        else:
            raise NotImplementedError

        return res

    def plot_ecops(self, style, data_type):

        assert self._in_vrfd_flag

        assert isinstance(style, str)
        assert style in self._poss_ans_stls

        poss_data_types = ['window', 'full']

        assert isinstance(data_type, str)
        assert data_type in poss_data_types

        assert self._in_vrfd_flag

        probs_arr = np.empty(
            (self._n_data_pts, self._ans_dims), dtype=np.float64, order='c')

        for i in range(self._ans_dims):
            probs_arr[:, i] = (
                rankdata(self._data_arr[:, i]) / (self._n_data_pts + 1))

        plt.figure(figsize=(10, 10))

        ttl = f'''
        PCA weights empirical copulas

        Analysis style: {style}
        Window type: {self._twt}
        Data Type: {data_type}
        {self._ans_dims} dimensions analyzed
        {self._n_uvecs:1.0E} unit vectors
        Peeling depth: {self._pl_dth}
        Spearman correlation: %0.4f
        Asymmetry 1: %1.0E
        Asymmetry 2: %1.0E
        Total steps: %d
        %d chull points
        '''

        emp_cop_out_dir = self._out_dir / 'empirical_copulas'
        emp_cop_out_dir.mkdir(exist_ok=True)

        for i in range(self._ans_dims):
            probs_arr_i = probs_arr[:, i]
            for j in range(self._ans_dims):
                if i >= j:
                    continue

                probs_arr_j = probs_arr[:, j]

                _idxs = self.get_boundary_point_idxs(style, data_type)

                if (style == 'peel') or (style == 'alt_peel'):

                    probs_i = probs_arr_i[~_idxs[0]]
                    probs_j = probs_arr_j[~_idxs[0]]

                    non_bd_i = probs_i[~_idxs[2]]
                    non_bd_j = probs_j[~_idxs[2]]

                    bd_i = probs_i[_idxs[2]]
                    bd_j = probs_j[_idxs[2]]

                elif style == 'raw':

                    probs_i = probs_arr_i
                    probs_j = probs_arr_j

                    non_bd_i = probs_arr_i[~_idxs[0]]
                    non_bd_j = probs_arr_j[~_idxs[0]]

                    bd_i = probs_arr_i[_idxs[0]]
                    bd_j = probs_arr_j[_idxs[0]]

                nchull_pts = bd_i.shape[0]

                correl = get_corrcoeff(probs_i, probs_j)

                asymms = get_asymms_sample(probs_i, probs_j)

                plt.scatter(
                    non_bd_i,
                    non_bd_j,
                    marker='o',
                    alpha=0.1,
                    label='non-bd pt',
                    color='C0')

                plt.scatter(
                    bd_i,
                    bd_j,
                    marker='o',
                    alpha=0.3,
                    label='bd pt',
                    color='C1')

                plt.xlabel(f'Dimension: {i}')
                plt.ylabel(f'Dimension: {j}')
                plt.legend(framealpha=0.5)

                cttl = ttl % (
                    correl,
                    asymms['asymm_1'],
                    asymms['asymm_2'],
                    self._n_data_pts,
                    nchull_pts)

                plt.title(
                    cttl,
                    fontdict={'ha': 'right'},
                    loc='right')

                plt.grid()

                out_fig_name = (
                    f'{data_type}_{style}_pca_wts_emp_cop_{i}_{j}.png')

                plt.savefig(
                    str(emp_cop_out_dir / out_fig_name), bbox_inches='tight')

                plt.clf()

        plt.close()

        return
