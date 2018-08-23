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

from .misc import ret_mp_idxs
from .analysis import AppearDisappearAnalysis
from .cyth import get_corrcoeff, get_asymms_sample

plt.ioff()


class AppearDisappearPlot:

    '''Plot the results of AppearDisappearAnalysis saved in the HDF5 file.'''

    def __init__(self, verbose=True):

        assert isinstance(verbose, bool)

        self.verbose = verbose

        adan = AppearDisappearAnalysis()

        self._h5_ds_names = adan._h5_ds_names

        self._var_labs_list = adan._var_labs_list

        self._poss_ans_stls = adan._poss_ans_stls

        self._dont_read_vars = (
            '_out_dir',  #  it is important to ignore this
            '_dn_flg',
            '_upld_bs_flg',
            '_pld_bs_flg',
            '_pld_upld_bs_flg',
            '_upld_pld_bs_flg')

        self._mp_pool = None
        self._n_cpus = None

        self._loo_flag = False

        self._h5_path_set_flag = False
        self._out_dir_set_flag = False
        self._fig_props_set_flag = False
        self._bs_vars_loaded_flag = False
        self._bef_plot_vars_set = False
        self._upld_pltd_flag = False
        self._pld_pltd_flag = False
        self._alt_pltd_flag = False
        self._plot_vrfd_flag = False
        return

    def set_hdf5(self, path):

        '''Set the path to inputs HDF5 file.

        Parameters
        ----------
        path : str or pathlib.Path
            Path to the input HDF5 file. This file is the same as the one
            written by the AppearDisappearAnalysis class.

        '''

        assert isinstance(path, (str, Path))

        path = Path(path).resolve()
        assert path.exists()
        assert path.is_file()

        self._h5_path = Path(path)

        if self.verbose:
            print('Set plotting input HDF5 path to:', str(path))

        self._h5_path_set_flag = True
        return

    def set_outputs_directory(self, out_dir, exist_ok=True):

        '''Set the directory to save all the plots in.

        Parameters
        ----------
        out_dir : str or pathlib.Path
            The directory in which to save the plots in. Will be created if
            not there.
        exist_ok : bool
            If False, raise an IOError (to avoid overwriting).
        '''

        assert isinstance(out_dir, (str, Path))

        out_dir = Path(out_dir).resolve()

        assert out_dir.parents[0].exists()

        out_dir.mkdir(exist_ok=exist_ok)

        self._out_dir = out_dir

        if self.verbose:
            print('Set plotting outputs path to:', str(out_dir))

        self._out_dir_set_flag = True
        return

    def set_fig_props(self, n_ticks, cmap):

        '''Set properties of the output figures.

        Parameters
        ----------
        n_ticks : int
            The number of ticks on both the axes. Default is 15 in case this
            function is not called.
        cmap : str or matplotlib.colors.Colormap
            The color map to use while plotting grids. Default is jet.
        '''

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

        '''Verify that all the inputs are correct.

        NOTE:
        -----
            These are just some additional checks. This function should
            always be called after all the inputs are set and ready.
        '''

        assert self._h5_path_set_flag
        assert self._out_dir_set_flag

        self._bef_plot()

        if self.verbose:
            print('All plotting inputs verfied to be correct.')

        self._plot_vrfd_flag = True
        return

    def set_n_cpus(self, n_cpus='auto'):

        # must call after verify to take effect
        '''Set the number of threads used while plotting.

        It is mainly required for the volume computation.

        Parameters
        ----------
        n_cpus : str, int
            Number of threads.
            If 'auto' then use one less than the maximum available threads.
        '''

        assert self._plot_vrfd_flag

        if n_cpus != 'auto':
            assert (n_cpus > 0) and np.isfinite(n_cpus)

        else:
            n_cpus = max(1, psutil.cpu_count() - 1)

        if self.verbose:
            print('Plotting N. cpus set to:', n_cpus)

        self._n_cpus = n_cpus
        return

    def plot_app_dis(self):

        '''Plot the appearing and disappearing cases on a grid.

        The types and number of plots are dependent on whatever was computed
        by the AppearDisappearAnalysis.cmpt_app_dis function.

        e.g. if analysis_style was 'peel' then both peeled and unpeeled cases
        are plotted.

        If bootstrapping was on then those results are also plotted.
        '''

        if self.verbose:
            print('Plotting appearing and disappearing cases...')

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

        if self.verbose:
            print('Done plotting appearing and disappearing cases.')

        return

    def plot_volumes(self, loo_flag=False):

        '''Plot the convex hull volumes of every window used in the analysis.

        Parameters
        ----------
        loo_flag : bool
            If True then leave one point out of all the points in the convex
            hull and compute and plot volume. This helps to detect points that
            are too far away and by removing them the hull volume drops
            significantly.
        '''

        assert self._ans_dims <= 6, (
            'More than 6D volume computation not supported!')

        assert isinstance(loo_flag, bool)

        self._loo_flag = loo_flag

        assert self._plot_vrfd_flag

        if self.verbose:
            print('Plotting moving window convex hull volumes...')
            print('Leave one-out flag:', loo_flag)

        self._plot_vols()

        if self.verbose:
            print('Done plotting moving window convex hull volumes.')
        return

    def plot_ecops(self):

        '''Plot empirical copulas of each dimension of the data array
        against the other.
        '''

        assert self._plot_vrfd_flag

        if self.verbose:
            print('Plotting empirical copulas of individual dimensions of '
                  'the input timeseries...')

        poss_data_types = ['window', 'full']

        h5_hdl = h5py.File(str(self._h5_path), mode='r')  # close it later
        bd_pts_gr = h5_hdl['bd_pts']

        emp_cop_out_dir = self._out_dir / 'empirical_copulas'
        emp_cop_out_dir.mkdir(exist_ok=True)

        for style in self._poss_ans_stls:
            for poss_data_type in poss_data_types:

                if style not in bd_pts_gr:
                    continue

                elif poss_data_type not in bd_pts_gr[style]:
                    continue

                self._plot_ecops(
                    style, poss_data_type, bd_pts_gr, emp_cop_out_dir)

        if self.verbose:
            print('Done plotting empirical copulas of individual '
                  'dimensions of the input timeseries.')

        return

    def plot_sim_anneal_opt(self):

        assert hasattr(self, '_sars'), 'sel_ortho_vecs_flag was off!'

        # optimization
        _, obj_ax = plt.subplots(figsize=(20, 10))
        acc_ax = obj_ax.twinx()

        plt.suptitle(
            f'Simulated annealing results for least correlated '
            f'vectors\' selection ({self._ans_dims} dimensions)')

        a1 = acc_ax.plot(
            self._sars,
            color='gray',
            alpha=0.5,
            label='acc_rate')
        p1 = obj_ax.plot(
            self._siovs,
            color='red',
            alpha=0.5,
            label='i_obj_val')

        p2 = obj_ax.plot(
            self._smovs,
            color='darkblue',
            alpha=0.5,
            label='min_obj_val')

        obj_ax.set_xlabel('Iteration No. (-)')
        obj_ax.set_ylabel('Objective function value (-)')
        acc_ax.set_ylabel('Acceptance rate (-)')

        obj_ax.grid()

        ps = p1 + p2 + a1
        lg_labs = [l.get_label() for l in ps]

        obj_ax.legend(ps, lg_labs, framealpha=0.5)

        plt.savefig(
            str(self._out_dir / 'sim_anneal.png'),
            bbox_inches='tight')
        plt.close()

        # correlation matrix
        plt.figure(figsize=(10, 10))
        plt.imshow(
            self._fca,
            origin='lower',
            vmin=0,
            vmax=1,
            cmap=plt.get_cmap('Blues'))

        for i in range(self._ans_dims):
            for j in range(self._ans_dims):
                plt.text(
                    i,
                    j,
                    f'{self._fca[i, j]: 0.4f}',
                    va='center',
                    ha='center')

        plt.xticks(np.arange(self._ans_dims), self._fidxs)
        plt.yticks(np.arange(self._ans_dims), self._fidxs)

        plt.xlabel('Selected index')
        plt.ylabel('Selected index')

        plt.title('Selected vectors\' abs. correlation matrix')

        plt.colorbar(label='abs. correlation', ticks=np.linspace(0, 1, 6))

        plt.savefig(
            str(self._out_dir / 'correlations.png'),
            bbox_inches='tight')
        plt.close()
        return

    def _plot_ecops(self, style, data_type, bd_pts_gr, emp_cop_out_dir):

        assert isinstance(style, str)
        assert isinstance(data_type, str)

        probs_arr = np.empty(
            (self._n_data_pts, self._ans_dims), dtype=np.float64, order='c')

        for i in range(self._ans_dims):
            probs_arr[:, i] = (
                rankdata(self._data_arr[:, i]) / (self._n_data_pts + 1))

        plt.figure(figsize=(10, 10))

        ttl = f'''
        Input data array's empirical copulas

        Analysis style: {style}
        Window type: {self._twt}
        Data Type: {data_type}
        {self._ans_dims} dimensions analyzed
        {self._n_uvecs:1.0E} unit vectors
        Peeling depth: {self._pl_dth}
        Spearman correlation: %0.4f
        Asymmetry 1: %1.3E
        Asymmetry 2: %1.3E
        Total steps: %d
        %d chull points
        '''

        for i in range(self._ans_dims):
            probs_arr_i = probs_arr[:, i]
            for j in range(self._ans_dims):
                if i >= j:
                    continue

                probs_arr_j = probs_arr[:, j]

                un_peel_idxs = bd_pts_gr[f'un_peel/{data_type}/idxs'][...]

                if (style == 'peel') or (style == 'alt_peel'):
                    peel_idxs = bd_pts_gr[f'peel/{data_type}/idxs'][...]

                    probs_i = probs_arr_i[~un_peel_idxs]
                    probs_j = probs_arr_j[~un_peel_idxs]

                    non_bd_i = probs_arr_i[(~peel_idxs) & (~un_peel_idxs)]
                    non_bd_j = probs_arr_j[(~peel_idxs) & (~un_peel_idxs)]

                    bd_i = probs_arr_i[peel_idxs]
                    bd_j = probs_arr_j[peel_idxs]

                elif style == 'un_peel':

                    probs_i = probs_arr_i
                    probs_j = probs_arr_j

                    non_bd_i = probs_arr_i[~un_peel_idxs]
                    non_bd_j = probs_arr_j[~un_peel_idxs]

                    bd_i = probs_arr_i[un_peel_idxs]
                    bd_j = probs_arr_j[un_peel_idxs]

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
                    probs_i.shape[0],
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

    def _bef_plot(self):

        if not self._fig_props_set_flag:
            self._nticks = 15
            self._cmap = plt.get_cmap('jet')

        h5_hdl = h5py.File(str(self._h5_path), mode='r')

        h5_dss_list = []

        for name in self._h5_ds_names:
            if name not in h5_hdl:
                continue

            h5_dss_list.append(h5_hdl[name])

        assert h5_dss_list

        n_dss = len(h5_dss_list)

        for i in range(n_dss):
            dss = h5_dss_list[i]
            var_labs = self._var_labs_list[i]

            for lab in var_labs:

                if lab in self._dont_read_vars:
                    continue

                if lab in dss:
                    setattr(self, lab, dss[lab][...])
                    setattr(getattr(self, lab).flags, 'writeable', False)

                elif lab in dss.attrs:
                    setattr(self, lab, dss.attrs[lab])

                elif lab in var_labs:
                    pass

                else:
                    raise KeyError(lab)

        h5_hdl.close()

        # conversions applied to some variables because hdf5 cant have them
        # in the format that is used here
        if (self._twt == 'month') or (self._twt == 'year'):

            self._t_idx = pd.to_datetime(self._t_idx, unit='s')

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

        '''Plot the appearing disappearing cases'''

        assert self._bef_plot_vars_set

        rats_arr = np.ma.masked_invalid(rats_arr)
        rats_bs_ul_arr = np.ma.masked_invalid(rats_bs_ul_arr)
        rats_bs_ll_arr = np.ma.masked_invalid(rats_bs_ll_arr)

        (rats_arr,
         rats_bs_ul_arr,
         rats_bs_ll_arr) = (
            rats_arr * 100,
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
        ax_l2.set_axis_off()

        cb1 = plt.colorbar(
            _ps1,
            ax=ax_l1,
            fraction=0.4,
            aspect=15,
            orientation='vertical')

        cb1.set_label('Percentage')

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

        '''Plot appearing disappearing cases along with bootstrapping
        results'''

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

    @staticmethod
    def _get_vols(step_idxs, args):

        '''Get volume of moving window convex hulls'''

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
        chull_idxs = np.unique(np.concatenate(chull_idxs))

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
                    self._loo_flag,
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

            res = None
            ress = None

            labs = np.concatenate(labs)
            vols = np.concatenate(vols)
            loo_vols = np.concatenate(loo_vols)
            n_chull_cts = np.concatenate(n_chull_cts)
            chull_idxs = np.concatenate(chull_idxs)

            chull_idxs = np.unique(chull_idxs)

        else:
            mp_cond = False

            dts_arr, = args

            args = (
                mp_cond,
                lab_cond,
                self._ans_dims,
                self._loo_flag,
                dts_arr,
                self._data_arr)

            (labs,
             vols,
             loo_vols,
             n_chull_cts,
             chull_idxs) = AppearDisappearPlot._get_vols(idxs_rng, args)

        return (labs, vols, loo_vols, n_chull_cts, chull_idxs)

    def _plot_vols(self):

        assert self._bef_plot_vars_set

        assert self._vdl

        if (self._n_cpus > 1) and (self._ans_dims >= 4):

            self._mp_pool = Pool(self._n_cpus)

        if self._mp_pool is not None:
            uvols_res = self._prep_for_vols(
                self._h5_path, '/dts_vars/_rudts', 'in_data/_data_arr')

        else:
            uvols_res = self._prep_for_vols(self._rudts)

        ulabs, uvols, uloo_vols, un_chull_cts, uchull_idxs = uvols_res

        plt_xs = np.arange(len(ulabs))

        vols_fig = plt.figure(figsize=(20, 7))
        npts_fig = plt.figure(figsize=(20, 7))
        bd_pt_fig = plt.figure(figsize=(40, 4))

        plt.figure(vols_fig.number)

        if self._loo_flag:
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

            if self._loo_flag:
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
        ttl_lab = 'Boundary points\' time series'

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
