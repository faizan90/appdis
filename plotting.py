'''
Created on Aug 8, 2018

@author: Faizan-Uni
'''
from pathlib import Path
from timeit import default_timer

import h5py
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.colors import Colormap
from matplotlib.cm import cmap_d
from scipy.stats import rankdata
from matplotlib.ticker import MaxNLocator

from .analysis import AppearDisappearAnalysis
from .cyth import get_corrcoeff, get_asymms_sample, get_2d_rel_hist

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
            print(3 * '\n', 50 * '#', sep='')
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
            print(3 * '\n', 50 * '#', sep='')
            print('Set plotting outputs path to:', str(out_dir))

        self._out_dir_set_flag = True
        return

    def set_fig_props(self, n_ticks, cmap, app_dis_cb_max):

        '''Set properties of the output figures.

        Parameters
        ----------
        n_ticks : int
            The number of ticks on both the axes. Default is 15 in case this
            function is not called.
        cmap : str or matplotlib.colors.Colormap
            The color map to use while plotting grids. Default is jet.
        app_dis_cb_max : int
            The maximum value on the appearing-disappearing figures. Should
            be between 0 and 100.
        '''

        assert isinstance(n_ticks, int)
        assert (n_ticks > 0) and np.isfinite(n_ticks)

        assert isinstance(cmap, (str, Colormap))

        assert isinstance(app_dis_cb_max, int)
        assert 0 < app_dis_cb_max <= 100

        self._n_ticks = n_ticks

        if isinstance(cmap, str):
            assert cmap in cmap_d
            self._cmap = plt.get_cmap(cmap)

        else:
            self._cmap = cmap

        self._adcm = app_dis_cb_max

        self._fig_props_set_flag = True
        return

    def verify(self):

        '''Verify that all the inputs are correct.

        NOTE
        ----
            These are just some additional checks. This function should
            always be called after all the inputs are set and ready.
        '''

        assert self._h5_path_set_flag
        assert self._out_dir_set_flag

        self._bef_plot()

        if self.verbose:
            print(3 * '\n', 50 * '#', sep='')
            print('All plotting inputs verified to be correct.')

        self._plot_vrfd_flag = True
        return

    def plot_app_dis(self):

        '''Plot the appearing and disappearing cases on a grid.

        The types and number of plots are dependent on whatever was computed
        by the AppearDisappearAnalysis.cmpt_app_dis function.

        e.g. if analysis_style was 'peel' then both peeled and unpeeled cases
        are plotted.

        If bootstrapping was on then those results are also plotted.
        '''

        begt = default_timer()

        if self.verbose:
            print(3 * '\n', 50 * '#', sep='')
            print('Plotting appearing and disappearing cases...')

        if self.verbose:
            print('Plotting unpeeled case...')

        self._plot_app_dis(self._upld, 'upld_plot.png', 'Both unpeeled')

        if (self._ans_stl == 'peel') or (self._ans_stl == 'alt_peel'):

            if self.verbose:
                print('Plotting peeled case...')

            self._plot_app_dis(self._pld, 'pld_plot.png', 'Both peeled')

            if self._ans_stl == 'alt_peel':

                if self.verbose:
                    print('Plotting peeled-unpeeled case...')

                self._plot_app_dis(
                    self._pld_upld,
                    'pld_upld_plot.png',
                    'Peeled-unpeeled')

                if self.verbose:
                    print('Plotting unpeeled-peeled case...')

                self._plot_app_dis(
                    self._upld_pld,
                    'upld_pld_plot.png',
                    'Unpeeled-peeled')

        if self._bs_flag:
            self._plot_bs()

        tott = default_timer() - begt

        if self.verbose:
            print(f'Done plotting appearing and disappearing cases in '
                  f'{tott:0.3f} secs.')

        return

    def plot_volumes(self):

        '''Plot the convex hull volumes of every window used in the analysis.
        '''

        assert self._ans_dims <= self._mvds, (
            f'More than {self._mvds}D volume computation not supported!')

        assert self._plot_vrfd_flag

        if self.verbose:
            print(3 * '\n', 50 * '#', sep='')
            print('Plotting moving window convex hull volumes...')
            print('Leave one-out flag:', self._loo_flag)

        begt = default_timer()

        self._plot_vols()

        tott = default_timer() - begt

        if self.verbose:
            print(f'Done plotting moving window convex hull volumes in '
                  f'{tott:0.3f} secs.')
        return

    def plot_ecops(self):

        '''Plot empirical copulas of each dimension of the data array
        against the other.
        '''

        assert self._plot_vrfd_flag

        if self.verbose:
            print(3 * '\n', 50 * '#', sep='')
            print('Plotting empirical copulas of individual dimensions of '
                  'the input timeseries...')

        begt = default_timer()

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

        tott = default_timer() - begt

        if self.verbose:
            print(f'Done plotting empirical copulas of individual dimensions '
                  f'of the input timeseries in {tott:0.3f} secs.')

        return

    def plot_sim_anneal_opt(self):

        assert hasattr(self, '_sars'), 'sel_ortho_vecs_flag was off!'

        if self.verbose:
            print(3 * '\n', 50 * '#', sep='')
            print('Plotting simulated annealing variables...')

        begt = default_timer()

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

        # optimization
        if self._n_data_dims != self._ans_dims:
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

        tott = default_timer() - begt

        if self.verbose:
            print(f'Done plotting simulated annealing variables in '
                  f'{tott:0.3f} secs.')
        return

    def _plot_ecops(self, style, data_type, bd_pts_gr, emp_cop_out_dir):

        assert isinstance(style, str)
        assert isinstance(data_type, str)

        probs_arr = np.empty(
            (self._n_data_pts, self._ans_dims), dtype=np.float64, order='c')

        for i in range(self._ans_dims):
            probs_arr[:, i] = (
                rankdata(self._data_arr[:, i]) / (self._n_data_pts + 1))

        plt.figure(figsize=(5, 5))

        ttl = f'''
        %s

        Analysis style: {style}
        Window type: {self._twt}
        Data Type: {data_type}
        Dimensions analyzed: {self._ans_dims}
        Unit vectors: {self._n_uvecs:1.0E}
        Peeling depth: {self._pl_dth}
        Spearman correlation: %0.4f
        Asymmetry 1: %1.3E
        Asymmetry 2: %1.3E
        Total steps: %d
        Chull points: %d
        '''

        # entropy stuff
        ent_bins = 10
        ent_ticks = np.arange(0.0, ent_bins + 0.1)
        ent_tick_labels = np.round(np.linspace(0.0, 1.0, ent_bins + 1), 1)

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
                    'Input data array\'s empirical copulas',
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
                    f'emp_cop_{data_type}_{style}_vec_{i}_{j}.png')

                plt.gca().set_aspect('equal', 'box')

                plt.savefig(
                    str(emp_cop_out_dir / out_fig_name), bbox_inches='tight')

                plt.clf()

                # entropy
                hist_2d = get_2d_rel_hist(probs_i, probs_j, ent_bins)
                ent_idxs = hist_2d > 0
                ent_grid = np.zeros((ent_bins, ent_bins))
                ent_grid[ent_idxs] = -(
                    hist_2d[ent_idxs] * np.log(hist_2d[ent_idxs]))

                ettl = ttl % (
                    'Input data array\'s empirical copula entropy',
                    correl,
                    asymms['asymm_1'],
                    asymms['asymm_2'],
                    probs_i.shape[0],
                    nchull_pts)

                plt.title(
                    ettl,
                    fontdict={'ha': 'right'},
                    loc='right')

                plt.pcolormesh(
                    ent_grid,
                    cmap=self._cmap)

                plt.xticks(ent_ticks, ent_tick_labels)
                plt.yticks(ent_ticks, ent_tick_labels)

                plt.xlabel(f'Dimension: {i} bins')
                plt.ylabel(f'Dimension: {j} bins')

                plt.colorbar(
                    label='Entropy', orientation='vertical', shrink=0.5)

                out_fig_name = (
                    f'entropy_{data_type}_{style}_vec_{i}_{j}.png')

                plt.gca().set_aspect('equal', 'box')

                plt.savefig(
                    str(emp_cop_out_dir / out_fig_name), bbox_inches='tight')

                plt.clf()

        plt.close()
        return

    def _bef_plot(self):

        if not self._fig_props_set_flag:
            self._nticks = 15
            self._cmap = plt.get_cmap('jet')
            self._adcm = 30

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
        plt.gca().set_aspect('equal', 'box')

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
            cmap=self._cmap,
            vmin=0,
            vmax=self._adcm)

        ttl = f'''
        Appearing and disappearing situations

        {arr_lab}
        Analysis style: {self._ans_stl}
        Window type: {self._twt}
        Dimensions analyzed: {self._ans_dims}
        Unit vectors: {self._n_uvecs:1.0E}
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

        '''Plot the appearing disappearing cases with boot-strapping'''

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

        rvmin = 0
        rvmax = self._adcm

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
        Dimensions analyzed: {self._ans_dims}
        Unit vectors: {self._n_uvecs:1.0E}
        Bootstraps: {self._n_bs}
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

        if self.verbose:
            print('Plotting unpeeled bootstrap case...')

        self._plot_bs_case(
            self._upld,
            self._upld_bs_ul,
            self._upld_bs_ll,
            'bs_upld_plot.png',
            'Both unpeeled (Bootstrap)')

        if (self._ans_stl == 'peel') or (self._ans_stl == 'alt_peel'):

            if self.verbose:
                print('Plotting peeled bootstrap case...')

            self._plot_bs_case(
                self._pld,
                self._pld_bs_ul,
                self._pld_bs_ll,
                'bs_pld_plot.png',
                'Both peeled (Bootstrap)')

            if self._ans_stl == 'alt_peel':
                if self.verbose:
                    print('Plotting peeled-unpeeled bootstrap case...')

                # case 1
                self._plot_bs_case(
                    self._pld_upld,
                    self._pld_upld_bs_ul,
                    self._pld_upld_bs_ll,
                    'bs_pld_upld_plot.png',
                    'Peeled-unpeeled (Bootstrap)')

                if self.verbose:
                    print('Plotting unpeeled-peeled bootstrap case...')

                # case 2
                self._plot_bs_case(
                    self._upld_pld,
                    self._upld_pld_bs_ul,
                    self._upld_pld_bs_ll,
                    'bs_upld_pld_plot.png',
                    'Unpeeled-peeled (Bootstrap)')
        return

    def _plot_vols(self):

        assert self._bef_plot_vars_set

        assert self._vdl

        h5_hdl = h5py.File(str(self._h5_path), mode='r')

        dss = h5_hdl['vol_boot_vars']

        _ulabs = dss['_ulabs']
        _uvols = dss['_uvols']
        _uloo_vols = dss['_uloo_vols']
        _un_chull_cts = dss['_un_chull_cts']
        _uchull_idxs = dss['_uchull_idxs']

        plt_xs = np.arange(len(_ulabs))

        vols_fig = plt.figure(figsize=(20, 7))
        npts_fig = plt.figure(figsize=(20, 7))

        plt.figure(vols_fig.number)

        if self._loo_flag:
            plt.scatter(
                _uloo_vols[:, 0],
                _uloo_vols[:, 1],
                marker='o',
                alpha=0.2,
                label='unpeeled leave-one',
                color='C0',
                zorder=6)

        plt.plot(
            plt_xs,
            _uvols,
            marker='o',
            alpha=0.6,
            label='unpeeled',
            color='C0',
            zorder=10)

        if 'min_vol_bs' in dss:
            min_vol_bs = dss['min_vol_bs']
            max_vol_bs = dss['max_vol_bs']

            plt.plot(
                min_vol_bs[:, 0],
                min_vol_bs[:, 1],
                alpha=0.6,
                ls=':',
                label='05% unpeeled',
                color='C2',
                zorder=8)

            plt.plot(
                max_vol_bs[:, 0],
                max_vol_bs[:, 1],
                alpha=0.6,
                ls=':',
                label='95% unpeeled',
                color='C3',
                zorder=8)

        plt.figure(npts_fig.number)
        plt.plot(
            plt_xs,
            _un_chull_cts,
            marker='o',
            alpha=0.6,
            label='unpeeled',
            color='C0',
            zorder=10)

        if (self._ans_stl == 'peel') or (self._ans_stl == 'alt_peel'):

            _pvols = dss['_pvols']
            _ploo_vols = dss['_ploo_vols']
            _pn_chull_cts = dss['_pn_chull_cts']
            _pchull_idxs = dss['_pchull_idxs']

            plt.figure(vols_fig.number)

            if self._loo_flag:
                plt.scatter(
                    _ploo_vols[:, 0],
                    _ploo_vols[:, 1],
                    marker='o',
                    alpha=0.2,
                    label='peeled leave-one',
                    color='C1',
                    zorder=6)

            plt.plot(
                plt_xs,
                _pvols,
                marker='o',
                alpha=0.6,
                label='peeled',
                color='C1',
                zorder=10)

            plt.figure(npts_fig.number)
            plt.plot(
                plt_xs,
                _pn_chull_cts,
                marker='o',
                alpha=0.6,
                label='peeled',
                color='C1',
                zorder=10)

        vol_corr = dss.attrs['_vbs_vol_corr']

        ttl = f'''
        %s

        Analysis style: {self._ans_stl}
        Window type: {self._twt}
        Dimensions analyzed: {self._ans_dims}
        Unit vectors: {self._n_uvecs:1.0E}
        Peeling depth: {self._pl_dth}
        Bootstraps: {self._n_vbs}
        Peeled-Unpeeled volume correlation: {vol_corr: 0.4f}
        Window size: {self._ws} {self._twt}(s)
        Starting, ending {self._twt}(s): {_ulabs[0]}, {_ulabs[-1]}
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
        tick_labs = _ulabs[::inc][:plt_xs.shape[0]]

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
        tick_labs = _ulabs[::inc][:plt_xs.shape[0]]

        plt.xticks(ticks, tick_labs, rotation=90)

        npts_fig.gca().yaxis.set_major_locator(MaxNLocator(integer=True))

        plt.grid()
        plt.legend()

        out_fig_name = 'chull_point_counts.png'

        plt.savefig(str(self._out_dir / out_fig_name), bbox_inches='tight')
        plt.close()

        h5_hdl.close()
        return

    def plot_ans_dims(self):

        _, axs = plt.subplots(
            nrows=self._ans_dims, ncols=1, figsize=(20, 8), sharex=True)

        ttl = f'''
        Analysed dimensions time series comparison

        Analysis style: {self._ans_stl}
        Window type: {self._twt}
        Dimensions analyzed: {self._ans_dims}
        Unit vectors: {self._n_uvecs:1.0E}
        Peeling depth: {self._pl_dth}
        Window size: {self._ws} {self._twt}(s)
        '''

        for i in range(self._ans_dims):
            ax = axs[i]

            ax.plot(
                self._data_arr[:, i],
                alpha=0.7,
                lw=0.5,
                label=f'Dim.: {i + 1:02d}')

            ax.grid()
            ax.legend(loc=1)

        ax.set_xlabel('Time step')

        axs[0].set_title(ttl, fontdict={'ha': 'right'}, loc='right')

        plt.savefig(
            str(self._out_dir / 'ans_dims_time_series.png'),
            bbox_inches='tight')

        plt.close()
        return
