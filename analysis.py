'''
@author: Faizan-Uni-Stuttgart

'''
from pathlib import Path

import h5py
import numpy as np
import pandas as pd
from monthdelta import monthmod

from depth_funcs import depth_ftn_mp as dftn

from .selection import AppearDisappearVectorSelection as ADVS
from .settings import AppearDisappearSettings as ADSS


class AppearDisappearAnalysis(ADVS, ADSS):

    '''Perform the appearing and disappearing events analysis

    Using Tukey's depth function and dividing the time series data into
    windows (N events per window), this class computes ratios of events
    that have appeared or disappeared for any two given event windows with
    respect to the test window.

    The time window can be a set of consecutive years or months. Events in
    test window are checked for containment inside the reference window.
    Points that have a depth of zero in the reference window are considered
    disappearing if the reference window is ahead of the test window in time
    and appearing if vice versa.

    For example, consider a dataset of 200 time steps (rows) and 2
    stations (columns). First 100 time steps are set as reference and the
    others as the test window. Using the Tukey's (or any) depth function,
    depth for each point of the test window in the reference window is
    computed. Tukey's depth funtion returns a zero for any point that is
    outside the convex hull (created by the points in the reference
    dataset). It returns a one if a point lies on the boundary of the convex
    hull. Let's say 10 points' depth is zero. So for this specific case,
    we have 10 appearing situations which is ten percent of the test
    window. This is the main output of this analysis. Based on the
    specified parameters, other outputs are also computed.

    Read the entire documentation for more information.
    '''

    def __init__(self, verbose=True, copy_input=False, mutability=False):

        ADVS.__init__(self, verbose, copy_input, mutability)
        ADSS.__init__(self, verbose)

        self._h5_hdl = None
        self._h5_path = None

        # Variables below are written to HDF5 file.
        # all labels must have a leading underscore.
        self._data_vars_labs = (
            '_data_arr',
            '_t_idx',
            '_t_idx_t',
            '_uvecs',
            '_n_data_pts',
            '_n_data_dims',
            '_n_uvecs',
            )

        self._sett_vars_labs = (
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
            '_vdl',
            )

        self._opt_vars_labs = (
            '_acorr_arr',
            '_irng',
            '_iat',
            '_tra',
            '_uaein',
            '_mis',
            '_mwocis',
            '_fidxs',
            '_fca',
            '_siovs',
            '_smovs',
            '_sars',
            )

        self._inter_vars_labs = (
            '_mwr',
            '_mwi',
            '_mss',
            )

        self._app_dis_vars_labs = (
            '_dn_flg',
            '_upld',
            '_pld',
            '_pld_upld',
            '_upld_pld',
            )

        self._dts_vars_labs = (
            '_rudts',
            '_rpdts',
            )

        self._boot_vars_labs = (
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
            )

        # sequence matters
        self._h5_ds_names = (
            'in_data',
            'settings',
            'vec_opt_vars',
            'inter_vars',
            'app_dis_arrs',
            'dts_vars',
            'boot_arrs',
            )

        self._var_labs_list = (
            self._data_vars_labs,
            self._sett_vars_labs,
            self._opt_vars_labs,
            self._inter_vars_labs,
            self._app_dis_vars_labs,
            self._dts_vars_labs,
            self._boot_vars_labs
            )

        self._rsm_hdf5_flag = False

        self._ann_vrfd_flag = False
        self._mw_rng_cmptd_flag = False
        self._app_dis_done_flag = False
        return

    def verify(self):

        '''Verify that all the inputs are correct.

        NOTE:
        -----
            These are just some additional checks. This function should be
            always called after all the inputs are set and ready.
        '''

        if not self._rsm_hdf5_flag:
            ADVS._AppearDisappearVectorSelection__verify(self)
            ADSS._AppearDisappearSettings__verify(self)

            assert self._ans_dims <= self._n_data_dims

            if self._t_idx_t == 'time':
                assert (self._twt == 'month') or (self._twt == 'year'), (
                    'Incompatible time_index and time_window_type!')

            elif self._t_idx_t == 'range':
                assert self._twt == 'range', (
                    'Incompatible time_index and time_window_type!')
                assert self._pl_dth < self._ws, (
                    'peel_depth cannot be greater than window_size '
                    'in this case!')

            else:
                raise NotImplementedError

            # this doesn't help much
            assert (self._ws + 1) < self._n_data_pts, (
                'window_size cannot be greater than the number of steps!')

        if self.verbose:
            print('All analysis inputs verified to be correct.')

        self._ann_vrfd_flag = True
        return

    def cmpt_appear_disappear(self):

        '''Perform the analysis after all the inputs are set and ready.'''

        self._bef_app_dis()

        assert self._ann_vrfd_flag, 'Call verify first!'

        if self.verbose:
            print('Computing appearing and disappearing cases...')

        pl_flg = (self._ans_stl == 'peel') or (self._ans_stl == 'alt_peel')

        cd_arr = self._data_arr[:, :self._ans_dims]

        # computations for un peeled case are kept here. The rest are
        # passed to other functions to avoid a long loop and too much white
        # space.

        for i in range(self._mwi):

            if np.all(self._dn_flg[i, :]):
                continue

            ris = (self._mwr >= i) & (self._mwr < (i + self._ws))
            if not ris.sum():
                continue

            if self.verbose:
                print('\n')
                _ridx = np.where(ris)[0]
                rbeg_time = self._t_idx[_ridx[+0]]
                rend_time = self._t_idx[_ridx[-1]]
                print('Reference begin and end time:', rbeg_time, rend_time)

            refr = cd_arr[ris, :].copy('c')

            if pl_flg or self._vdl:
                refr_refr_dts = self._get_dts(refr, refr)
                refr_pldis = refr_refr_dts > self._pl_dth
                refr_pld = refr[refr_pldis, :]

                if self._vdl:
                    ct = refr_refr_dts.shape[0]
                    self._rudts['cts'][i] = ct
                    self._rudts['dts'][i, :ct] = refr_refr_dts
                    self._rudts['idx'][i, :ct] = np.where(ris)[0]

                    step_lab = ''

                    if self._twt == 'year':
                        step_lab = int(self._t_idx[ris][0].strftime('%Y'))

                    elif self._twt == 'month':
                        step_lab = int(self._t_idx[ris][0].strftime('%Y%m'))

                    elif self._twt == 'range':
                        step_lab = self._t_idx[ris][0]

                    self._rudts['lab'][i] = step_lab

                    if pl_flg:
                        refr_pld_dts = self._get_dts(refr_pld, refr_pld)
                        pct = refr_pld_dts.shape[0]
                        self._rpdts['cts'][i] = pct
                        self._rpdts['dts'][i, :pct] = refr_pld_dts
                        self._rpdts['idx'][i, :pct] = (
                            np.where(ris)[0][refr_pldis])
                        self._rpdts['lab'][i] = self._rudts['lab'][i]

                if self._bs_flag:
                    rpis = np.zeros_like(ris, dtype=bool)
                    rpis = ris | rpis
                    rpis[ris] = refr_pldis

            for j in range(self._mwi):

                if self._dn_flg[i, j]:
                    continue

                if i == j:
                    self._set_to_zero(i, pl_flg)
                    continue

                tis = (self._mwr >= j) & (self._mwr < (j + self._ws))
                if not tis.sum():
                    continue

                if self.verbose:
                    _tidx = np.where(tis)[0]
                    tbeg_time = self._t_idx[_tidx[+0]]
                    tend_time = self._t_idx[_tidx[-1]]
                    print('\tTest begin and end time:', tbeg_time, tend_time)

                test = cd_arr[tis, :].copy('c')

                self._fill_get_rat(refr, test, i, j, self._upld)

                if self._bs_flag:
                    self._cmpt_bs_lims(
                        ris,
                        tis,
                        cd_arr,
                        i,
                        j,
                        self._upld_bs_ul,
                        self._upld_bs_ll,
                        self._upld_bs_flg)

                if pl_flg:
                    if not self._bs_flag:
                        args = []

                    else:
                        args = [ris, tis, rpis, cd_arr]

                    self._pld_upld_rats(refr, test, refr_pld, i, j, *args)

                if self._fh_flag == 2:
                    self._ut_hdf5()

                # after the _ut_hdf5 call because it could have broken
                # during updating
                self._dn_flg[i, j] = True

            if self._fh_flag == 1:
                self._ut_hdf5()

        self._app_dis_done_flag = True
        self._aft_app_dis()

        if self.verbose:
            print('Done computing appearing and disappearing cases.')
        return

    def resume_from_hdf5(self, path):

        '''
        Resume computations from the state saved in the HDF5

        Parameters
        ----------
        path : str, pathlib.Path
            Path to HDF5 file
        '''

        assert isinstance(path, (str, Path)), (
            'path not an instance of str of pathlib.Path!')

        path = Path(path).resolve()

        assert path.exists(), 'Input file not found!'
        assert path.is_file(), 'Input is not a file!'

        self._h5_hdl = h5py.File(str(path), driver='core', mode='a')

        h5_dss_list = []

        for name in self._h5_ds_names:
            if name not in self._h5_hdl:
                continue

            h5_dss_list.append(self._h5_hdl[name])

        assert h5_dss_list, (
            'The given file has no variables that match the ones needed '
            'here!')

        n_dss = len(h5_dss_list)

        for i in range(n_dss):
            dss = h5_dss_list[i]
            var_labs = self._var_labs_list[i]

            for lab in var_labs:
                if lab in dss:
                    setattr(self, lab, dss[lab][...])

                elif lab in dss.attrs:
                    setattr(self, lab, dss.attrs[lab])

                elif lab in var_labs:
                    pass

                else:
                    raise KeyError(f'Unknown variable: {lab}')

        # conversions applied to some variables because hdf5 can't have them
        # in the format that is used here.
        if (self._twt == 'month') or (self._twt == 'year'):

            self._t_idx = pd.to_datetime(self._t_idx, unit='s')

        self._out_dir = Path(self._out_dir)

        self._rsm_hdf5_flag = True
        self.verify()

        if self.verbose:
            print('Loaded data from HDF5.')
            not_cmptd_idxs = (np.isnan(self._upld.ravel())).sum()
            tot_idxs = self._upld.ravel().shape[0]
            print(f'{not_cmptd_idxs} steps out of {tot_idxs} to go!')

        self.cmpt_appear_disappear()
        return

    def terminate_analysis(self):

        '''Finish unfinished business here'''

        assert self._app_dis_done_flag

        if self._hdf5_flag:

            self._h5_hdl.close()

            self._h5_hdl = None
        return

    def get_moving_window_range(self):

        assert self._mw_rng_cmptd_flag, 'Moving window range not computed!'

        return self._mwr

    def get_number_of_windows(self):

        assert self._mw_rng_cmptd_flag, 'Moving window range not computed!'

        return self._mwi

    def get_maximum_steps_per_window(self):

        assert self._mw_rng_cmptd_flag, 'Moving window range not computed!'

        return self._mss

    def get_unpeeled_appear_disappear_ratios(self):

        assert self._app_dis_done_flag, 'Call cmpt_appear_disappear first!'

        return self._upld

    def get_peeled_appear_disappear_ratios(self):

        assert self._app_dis_done_flag, 'Call cmpt_appear_disappear first!'
        assert (self._ans_stl == 'peel') or (self._ans_stl == 'alt_peel'), (
            'Incompatible analysis style!')

        return self._pld

    def get_alternating_appear_disappear_ratios(self):

        assert self._app_dis_done_flag, 'Call cmpt_appear_disappear first!'
        assert self._ans_stl == 'alt_peel', 'Incompatible analysis style!'

        return (self._pld_upld, self._upld_pld)

    def _cmpt_mw_rng(self):

        '''Compute moving window range, number of possible windows and
        number of possible maximum steps per window'''

        win_rng = []

        if self._twt == 'month':
            t_idx_0 = self._t_idx[0]

            for date in self._t_idx:
                win_rng.append(monthmod(t_idx_0, date)[0].months)

        elif self._twt == 'year':
            t_idx_0 = self._t_idx[0].year

            for date in self._t_idx:
                win_rng.append(date.year - t_idx_0)

        elif self._twt == 'range':
            win_rng = np.arange(self._n_data_pts, dtype=np.int64, order='c')

        else:
            raise NotImplementedError

        win_rng = np.array(win_rng, dtype=np.int64, order='c')
        win_rng.flags.writeable = False

        max_val = win_rng.max()

        assert np.all(win_rng >= 0), 'time_index is not ascending!'
        assert max_val > self._ws, (
            'Number of possible windows less than window_size!')

        if (self._twt == 'month') or (self._twt == 'year'):

            unq_vals = np.unique(win_rng)
            mwi = unq_vals.shape[0] - self._ws

        elif self._twt == 'range':
            mwi = win_rng.shape[0] - self._ws
            # max ct no implemented yet!

        self._mwr = win_rng
        self._mwi = mwi + 1

        assert self._mwi > 1, (
            'Number of final windows cannot be less than two!')

        max_steps = 0
        for i in range(self._mwi):
            ris = (self._mwr >= i) & (self._mwr < (i + self._ws))
            max_steps = max(max_steps, ris.sum())

        max_steps = int(max_steps)
        assert max_steps, 'This should not happen!'

        self._mss = max_steps

        self._mw_rng_cmptd_flag = True
        return

    def _bef_app_dis(self):

        '''Initiate all required variables before analysis'''

        assert self._ann_vrfd_flag, 'Call verify first!'

        if self._rsm_hdf5_flag:
            return

        if self._orth_vecs_flag:
            if self.verbose:
                print('Finding least correlated vectors...')

            self.generate_vector_indicies_set()

            if self.verbose:
                print('Done finding least correlated vectors.')

            sel_idxs = self.get_final_vector_indicies().tolist()

            not_sel_idxs = []
            for i in range(self._n_data_dims):
                if i in sel_idxs:
                    continue

                not_sel_idxs.append(i)

            fin_idxs = sel_idxs + not_sel_idxs

            assert len(fin_idxs) == self._n_data_dims
            assert np.unique(fin_idxs).shape[0] == self._n_data_dims

            self._data_arr = self._data_arr[:, fin_idxs].copy('c')
            self._data_arr.flags.writeable = self._mtbl_flag

        self._cmpt_mw_rng()
        assert self._mw_rng_cmptd_flag, 'Moving window range not computed!'

        # ratio of test points outside refr
        self._upld = np.full(
            (self._mwi, self._mwi), np.nan, dtype=np.float64, order='c')

        self._dn_flg = np.zeros_like(self._upld, dtype=bool)

        if self._vdl:

            # these 3 should be copied for all
            vol_dt = np.dtype([
                ('cts', np.int64, (self._mwi,)),
                ('lab', np.int64, (self._mwi,)),
                ('idx', np.int64, (self._mwi, self._mss)),
                ('dts', np.int64, (self._mwi, self._mss))])

            vol_cts = np.zeros(self._mwi, dtype=np.int64, order='c')
            vol_lab = vol_cts.copy()
            vol_idx = np.zeros(
                (self._mwi, self._mss), dtype=np.int64, order='c')
            vol_dts = vol_idx.copy()

            self._rudts = np.array(
                (vol_cts, vol_lab, vol_idx, vol_dts), dtype=vol_dt, order='c')

        if self._bs_flag:
            self._upld_bs_ul = np.full(self._upld.shape, -np.inf)
            self._upld_bs_ll = np.full(self._upld.shape, +np.inf)

            self._upld_bs_flg = np.zeros_like(
                self._upld, dtype=bool, order='c')

        if (self._ans_stl == 'peel') or (self._ans_stl == 'alt_peel'):

            self._pld = self._upld.copy()

            self._rpdts = self._rudts.copy()

            if self._bs_flag:
                self._pld_bs_ul = np.full(self._pld.shape, -np.inf)
                self._pld_bs_ll = np.full(self._pld.shape, +np.inf)

                self._pld_bs_flg = np.zeros_like(
                    self._pld, dtype=bool, order='c')

            if self._ans_stl == 'alt_peel':
                # in the name, first position is for refr, the second for test
                self._pld_upld = self._pld.copy()
                self._upld_pld = self._pld_upld.copy()

                if self._bs_flag:
                    self._pld_upld_bs_ul = np.full(self._pld.shape, -np.inf)
                    self._pld_upld_bs_ll = np.full(self._pld.shape, +np.inf)

                    self._upld_pld_bs_ul = np.full(
                        self._pld_upld.shape, -np.inf)
                    self._upld_pld_bs_ll = np.full(
                        self._pld_upld.shape, +np.inf)

                    self._pld_upld_bs_flg = np.zeros_like(
                        self._pld_upld, dtype=bool, order='c')

                    self._upld_pld_bs_flg = np.zeros_like(
                        self._upld_pld, dtype=bool, order='c')

        if self._hdf5_flag:
            self._init_hdf5_ds()
        return

    def _init_hdf5_ds(self):

        '''Initialize the outputs HDF5 file and write the appropriate
        variables to it.
        '''

        self._h5_path = self._out_dir / 'app_dis_ds.hdf5'
        self._h5_hdl = h5py.File(
            str(self._h5_path), mode='w', driver='core')

        # sequence matters
        h5_dss_list = list(self._h5_ds_names[:5])

        if self._vdl:
            h5_dss_list.append('dts_vars')

        if self._bs_flag:
            h5_dss_list.append('boot_arrs')

        assert h5_dss_list, 'No variables selected for writing to HDF5!'

        n_dss = len(h5_dss_list)

        for i in range(n_dss):
            dss = self._h5_hdl.create_group(h5_dss_list[i])
            var_labs = self._var_labs_list[i]

            for lab in var_labs:

                try:
                    var = getattr(self, lab)

                except AttributeError:
                    continue

                if isinstance(var, np.ndarray):
                    dss[lab] = var

                elif isinstance(var, (str, int, float)):
                    dss.attrs[lab] = var

                elif isinstance(var, Path):
                    dss.attrs[lab] = str(var)

                elif ((self._twt == 'month') or
                      (self._twt == 'year')) and (lab == '_t_idx'):

                    _td = pd.Timedelta('1s')
                    _min_t = pd.Timestamp("1970-01-01")

                    dss[lab] = (self._t_idx - _min_t) // _td

                else:
                    raise KeyError(
                        f'Don\'t know how to handle the variable {lab} of '
                        f'type {type(var)}')

        self._h5_hdl.flush()
        return

    def _aft_app_dis(self):

        '''Things to do write after the analysis finishes'''

        assert self._app_dis_done_flag

        if self._fh_flag == 0:
            self._ut_hdf5()

        if (self._ans_stl == 'peel') or (self._ans_stl == 'alt_peel'):

            self._save_boundary_point_idxs('peel', 'full')
            self._save_boundary_point_idxs('peel', 'window')

        else:
            self._save_boundary_point_idxs('un_peel', 'full')

        self._save_boundary_point_idxs('un_peel', 'window')

        return

    def _ut_hdf5(self):

        '''Flush variables to the HDF5 file.'''

        if not self._hdf5_flag:
            return

        rds = self._h5_hdl['app_dis_arrs']

        for rd in rds.keys():
            exec(f'rds[\'{rd}\'][...] = self.{rd}')

        if self._vdl:
            vds = self._h5_hdl['dts_vars']

            for vd in vds.keys():
                exec(f'vds[\'{vd}\'][...] = self.{vd}')

        if self._bs_flag:
            bsds = self._h5_hdl['boot_arrs']

            for bsd in bsds.keys():
                exec(f'bsds[\'{bsd}\'][...] = self.{bsd}')

        self._h5_hdl.flush()
        return

    def _get_dts(self, refr, test):

        '''Get depths of test in refr'''

        if refr.shape[0] and test.shape[0]:
            dts = dftn(refr, test, self._uvecs, self._n_cpus)

        else:
            dts = np.array([], dtype=np.int64)

        return dts

    def _fill_get_rat(self, refr, test, idx_i, idx_j, farr):

        '''Compute the ratio of points in test that are outside of refr and
        put it in farr.

        Return the depth of all the test points as well.
        '''

        test_dts = self._get_dts(refr, test)

        if test_dts.shape[0]:
            farr[idx_i, idx_j] = (test_dts == 0).sum() / test_dts.shape[0]

        return test_dts

    def _fill_get_rat_bs(self, refr, test, idx_i, idx_j, farr_ul, farr_ll, farr_flgs):

        '''Compute the ratio of points in test that are outside of refr and
        update the ratios in the farr_ul and farr_ll (this is for
        bootstrapping).

        Return the depth of all the test points as well.
        '''

        test_dts = self._get_dts(refr, test)

        farr_flgs[idx_i, idx_j] = True

        if test_dts.shape[0]:
            rat = (test_dts == 0).sum() / test_dts.shape[0]

            farr_ul[idx_i, idx_j] = max(farr_ul[idx_i, idx_j], rat)
            farr_ll[idx_i, idx_j] = min(farr_ll[idx_i, idx_j], rat)

        return test_dts

    def _pld_upld_rats(self, refr, test, refr_pld, idx_i, idx_j, *args):

        '''Just to have less white space'''

        test_test_dts = self._get_dts(test, test)
        test_pldis = test_test_dts > self._pl_dth
        test_pld = test[test_pldis, :]

        self._fill_get_rat(refr_pld, test_pld, idx_i, idx_j, self._pld)

        if self._bs_flag:
            ris, tis, rpis, cd_arr = args[0], args[1], args[2], args[3]

            tpis = np.zeros_like(tis, dtype=bool)
            tpis = tis | tpis
            tpis[tis] = test_pldis

            self._cmpt_bs_lims(
                rpis,
                tpis,
                cd_arr,
                idx_i,
                idx_j,
                self._pld_bs_ul,
                self._pld_bs_ll,
                self._pld_bs_flg)

        if self._ans_stl == 'alt_peel':
            self._fill_get_rat(refr_pld, test, idx_i, idx_j, self._pld_upld)
            self._fill_get_rat(refr, test_pld, idx_i, idx_j, self._upld_pld)

            if self._bs_flag:
                self._cmpt_bs_lims(
                    rpis,
                    tis,
                    cd_arr,
                    idx_i,
                    idx_j,
                    self._pld_upld_bs_ul,
                    self._pld_upld_bs_ll,
                    self._pld_upld_bs_flg)

                self._cmpt_bs_lims(
                    ris,
                    tpis,
                    cd_arr,
                    idx_i,
                    idx_j,
                    self._upld_pld_bs_ul,
                    self._upld_pld_bs_ll,
                    self._upld_pld_bs_flg)

        return

    def _cmpt_bs_lims(
            self,
            ris,
            tis,
            cd_arr,
            idx_i,
            idx_j,
            farr_ul,
            farr_ll,
            farr_flgs):

        '''Compute upper and lower bounds of ratios that are appearing or
        disappearing using bootstrapping.
        '''

        if farr_flgs[idx_j, idx_i]:
            farr_ul[idx_i, idx_j] = farr_ul[idx_j, idx_i]
            farr_ll[idx_i, idx_j] = farr_ll[idx_j, idx_i]
            farr_flgs[idx_i, idx_j] = farr_flgs[idx_j, idx_i]
            return

        rmwr = self._mwr[ris]
        tmwr = self._mwr[tis]
        rtmwrs = ris | tis

        n_refr = rmwr.shape[0]

        n_rbsis = np.unique(rmwr).shape[0] + np.unique(tmwr).shape[0]

        unacc_rseq = []
        for wi in rmwr:
            if wi in unacc_rseq:
                continue

            unacc_rseq.append(wi)

        unacc_tseq = []
        for wi in tmwr:
            if wi in unacc_tseq:
                continue

            unacc_tseq.append(wi)

        unacc_seq = unacc_rseq + unacc_tseq

        unq_rtmwrs = np.unique(np.concatenate((rmwr, tmwr)))

        bs_ctr = 0
        unacc_seq_ctr = 0
        while bs_ctr < self._n_bs:

            if unacc_seq_ctr >= 5:
                raise ValueError(
                    'Window size is too short for having unique sequences '
                    'for bootstrapping. Increase window size!')

            rbsis = np.random.choice(
                unq_rtmwrs,
                size=n_rbsis,
                replace=True).tolist()

            if np.all(unacc_seq == rbsis):
                unacc_seq_ctr += 1

                if self.verbose:
                    print('Unacceptable sequence encountered in '
                          'bootstrapping!')
                continue

            bs_set = []
            for ibs in rbsis:
                ibsis = (self._mwr == ibs) & rtmwrs
                bs_set.append(cd_arr[ibsis, :])

            bs_set = np.concatenate(bs_set, axis=0)

            refr_bs = bs_set[:n_refr, :]
            test_bs = bs_set[n_refr:, :]

            self._fill_get_rat_bs(
                refr_bs, test_bs, idx_i, idx_j, farr_ul, farr_ll, farr_flgs)
            bs_ctr += 1
        return

    def _save_boundary_point_idxs(self, style, data_type):

        assert self._ann_vrfd_flag

        assert isinstance(style, str)
        assert style in self._poss_ans_stls

        poss_data_types = ['window', 'full']

        assert isinstance(data_type, str)
        assert data_type in poss_data_types

        if not self._hdf5_flag:
            return

        grp_name = 'bd_pts'

        _td = pd.Timedelta('1s')
        _min_t = pd.Timestamp("1970-01-01")

        if grp_name not in self._h5_hdl:
            bd_pts_gr = self._h5_hdl.create_group(grp_name)

        else:
            bd_pts_gr = self._h5_hdl[grp_name]

        if data_type == 'window':
            if style == 'un_peel':
                dts_arr = self._rudts

            elif style == 'peel':

                assert (
                    (self._ans_stl == 'peel') or
                    (self._ans_stl == 'alt_peel'))

                dts_arr = self._rpdts

            else:
                raise NotImplementedError

            res = self._get_win_bd_idxs(dts_arr)
            bd_pts_gr[f'{style}/{data_type}/idxs'] = res[0]
            bd_pts_gr[f'{style}/{data_type}/time'] = (res[1] - _min_t) // _td

        elif data_type == 'full':
            res = self._get_full_bd_idxs(style)

            bd_pts_gr[f'un_peel/{data_type}/idxs'] = res[0]
            bd_pts_gr[f'un_peel/{data_type}/time'] = (res[1] - _min_t) // _td

            if style == 'peel':
                bd_pts_gr[f'peel/{data_type}/idxs'] = res[2]
                bd_pts_gr[f'peel/{data_type}/time'] = (res[3] - _min_t) // _td

        else:
            raise NotImplementedError

        self._h5_hdl.flush()
        return

    def _get_win_bd_idxs(self, dts_arr):

        chull_idxs = []
        idxs_rng = np.arange(self._mwi)

        dth = max(1, self._pl_dth)  # at zero there are no bd_pts

        for i in idxs_rng:
            ct = dts_arr['cts'][i]

            dts = dts_arr['dts'][i, :ct]
            idxs = dts_arr['idx'][i, :ct]

            bd_idxs = idxs[dts <= dth]

            chull_idxs.append(bd_idxs)

        _ = np.unique(np.concatenate(chull_idxs))

        chull_idxs = np.zeros(self._n_data_pts, dtype=bool)
        chull_idxs[_] = True
        chull_time_idxs = self._t_idx[chull_idxs]

        return (chull_idxs, chull_time_idxs)

    def _get_full_bd_idxs(self, style):

        assert (style == 'peel') or (style == 'un_peel')

        cd_arr = self._data_arr[:, :self._ans_dims].copy('c')

        refr_refr_dts = dftn(cd_arr, cd_arr, self._uvecs, self._n_cpus)

        dth = max(1, self._pl_dth)  # at zero there are no bd_pts

        upld_chull_idxs = refr_refr_dts <= dth
        upld_chull_time_idxs = self._t_idx[upld_chull_idxs]

        ret = [upld_chull_idxs, upld_chull_time_idxs, ]

        if style == 'peel':
            refr_pld = cd_arr[~upld_chull_idxs].copy('c')

            refr_refr_pld_dts = dftn(
                refr_pld, refr_pld, self._uvecs, self._n_cpus)

            pld_chull_idxs = np.zeros(self._n_data_pts, dtype=bool)
            pld_chull_idxs[~upld_chull_idxs] = (
                refr_refr_pld_dts <= dth)

            pld_chull_time_idxs = self._t_idx[pld_chull_idxs]

            ret += [pld_chull_idxs, pld_chull_time_idxs]

        return ret

    def _set_to_zero(self, i, pl_flg):
        self._upld[i, i] = 0

        if pl_flg:
            self._pld[i, i] = 0

            if self._ans_stl == 'alt_peel':
                self._pld_upld[i, i] = 0
                self._upld_pld[i, i] = 0

        if self._bs_flag:
            self._upld_bs_ul[i, i] = 0
            self._upld_bs_ll[i, i] = 0

            if pl_flg:
                self._pld_bs_ul[i, i] = 0
                self._pld_bs_ll[i, i] = 0

                if self._ans_stl == 'alt_peel':
                    self._pld_upld_bs_ul[i, i] = 0
                    self._pld_upld_bs_ll[i, i] = 0

                    self._upld_pld_bs_ul[i, i] = 0
                    self._upld_pld_bs_ll[i, i] = 0
        return
