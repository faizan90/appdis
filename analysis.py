'''
@author: Faizan-Uni-Stuttgart

'''
from pathlib import Path

import h5py
import numpy as np
import pandas as pd
from monthdelta import monthmod

from depth_funcs import depth_ftn_mp as dftn

from .data import AppearDisappearData
from .settings import AppearDisappearSettings


class AppearDisappearAnalysis:

    def __init__(self, verbose=True):

        # we could inherit but this seems to be more flexible

        assert isinstance(verbose, bool)

        self.verbose = verbose

        self._h5_hdl = None
        self._h5_path = None

        # all labels must have a leading underscore
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
            '_rdts',
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
        self.h5_ds_names = (
            'in_data',
            'settings',
            'inter_vars',
            'app_dis_arrs',
            'dts_vars',
            'boot_arrs',
            )

        self.var_labs_list = (
            self._data_vars_labs,
            self._sett_vars_labs,
            self._inter_vars_labs,
            self._app_dis_vars_labs,
            self._dts_vars_labs,
            self._boot_vars_labs
            )

        self._data_set_flag = False
        self._sett_set_flag = False
        self._rsm_hdf5_flag = False

        self._in_vrfd_flag = False
        self._mw_rng_cmptd_flag = False
        return

    def set_data(self, appear_disappear_data_obj):

        assert isinstance(appear_disappear_data_obj, AppearDisappearData)
        assert appear_disappear_data_obj._in_vrfd_flag

        addo = appear_disappear_data_obj

        for v in self._data_vars_labs:
            setattr(self, v, getattr(addo, v))

        self._data_set_flag = True
        return

    def set_settings(self, appear_disappear_settings_obj):

        assert isinstance(
            appear_disappear_settings_obj, AppearDisappearSettings)
        assert appear_disappear_settings_obj._in_vrfd_flag

        adso = appear_disappear_settings_obj

        for v in self._sett_vars_labs:
            setattr(self, v, getattr(adso, v))

        self._sett_set_flag = True
        return

    def resume_from_hdf5(self, path):

        assert isinstance(path, (str, Path))

        path = Path(path).resolve()

        assert path.exists()
        assert path.is_file()

        self._h5_hdl = h5py.File(str(path), driver='core', mode='a')

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

        # conversions applied to some variables because hdf5 cant have them
        # in the format that is used here
        if (self._twt == 'month') or (self._twt == 'year'):

            self._t_idx = pd.to_datetime(self._t_idx, unit='s')

        self._out_dir = Path(self._out_dir)

        self._rsm_hdf5_flag = True
        self.verify()
        return

    def verify(self):

        if not self._rsm_hdf5_flag:
            assert self._data_set_flag
            assert self._sett_set_flag

            if self._t_idx_t == 'time':
                assert (self._twt == 'month') or (self._twt == 'year')

            elif self._t_idx_t == 'range':
                assert self._twt == 'range'
                assert self._pl_dth < self._ws

            else:
                raise NotImplementedError

            # this doesn't help much
            assert self._ws < self._n_data_pts
            assert (self._ws + 1) < self._n_data_pts

        self._in_vrfd_flag = True
        return

    def cmpt_appear_disappear(self):

        self._bef_app_dis()

        assert self._in_vrfd_flag

        pl_flg = (self._ans_stl == 'peel') or (self._ans_stl == 'alt_peel')

        cd_arr = self._data_arr[:, :self._ans_dims]

        # computations for _ans_stl == 'raw' are kept here. The rest are
        # passed to other functions to avoid a long loop and too much white
        # space

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
                    self._rdts['cts'][i] = ct
                    self._rdts['dts'][i, :ct] = refr_refr_dts
                    self._rdts['idx'][i, :ct] = np.where(ris)[0]

                    step_lab = ''

                    if self._twt == 'year':
                        step_lab = int(self._t_idx[ris][0].strftime('%Y'))

                    elif self._twt == 'month':
                        step_lab = int(self._t_idx[ris][0].strftime('%Y%m'))

                    elif self._twt == 'range':
                        step_lab = self._t_idx[ris][0]

                    self._rdts['lab'][i] = step_lab

                if self._bs_flag:
                    rpis = np.zeros_like(ris, dtype=bool)
                    rpis = ris | rpis
                    rpis[ris] = refr_pldis

            for j in range(self._mwi):

                if self._dn_flg[i, j]:
                    continue

                if i == j:
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
                        self._pld_upld_rats(refr, test, refr_pld, i, j)

                    else:
                        self._pld_upld_rats(
                            refr,
                            test,
                            refr_pld,
                            i,
                            j,
                            ris,
                            tis,
                            rpis,
                            cd_arr)

                if self._fh_flag == 2:
                    self._ut_hdf5()

                # after the update call because it could have broken
                # during updating
                self._dn_flg[i, j] = True

            if self._fh_flag == 1:
                self._ut_hdf5()

        self._aft_app_dis()
        return

    def terminate_analysis(self):

        if self._hdf5_flag:
            self._ut_hdf5()
            self._h5_hdl.close()

        self._h5_hdl = None
        return

    def _cmpt_mw_rng(self):

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

        assert np.all(win_rng >= 0)
        assert max_val > self._ws

        if (self._twt == 'month') or (self._twt == 'year'):

            unq_vals = np.unique(win_rng)
            mwi = unq_vals.shape[0] - self._ws

        elif self._twt == 'range':
            mwi = win_rng.shape[0] - self._ws
            # max ct no implemented yet!

        self._mwr = win_rng
        self._mwi = mwi + 1

        assert self._mwi > 1

        max_steps = 0
        for i in range(self._mwi):
            ris = (self._mwr >= i) & (self._mwr < (i + self._ws))
            max_steps = max(max_steps, ris.sum())

        max_steps = int(max_steps)
        assert max_steps

        self._mss = max_steps

        self._mw_rng_cmptd_flag = True
        return

    def _bef_app_dis(self):

        assert self._in_vrfd_flag

        if self._rsm_hdf5_flag:
            return

        self._cmpt_mw_rng()
        assert self._mw_rng_cmptd_flag

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

            self._rdts = np.array(
                (vol_cts, vol_lab, vol_idx, vol_dts), dtype=vol_dt, order='c')

        if self._bs_flag:
            self._upld_bs_ul = np.full(self._upld.shape, -np.inf)
            self._upld_bs_ll = np.full(self._upld.shape, +np.inf)

            self._upld_bs_flg = np.zeros_like(
                self._upld, dtype=bool, order='c')

        if (self._ans_stl == 'peel') or (self._ans_stl == 'alt_peel'):

            self._pld = self._upld.copy()

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
        self._h5_path = self._out_dir / 'app_dis_ds.hdf5'
        self._h5_hdl = h5py.File(
            str(self._h5_path), mode='w', driver='core')

        # sequence matters
        h5_dss_list = list(self.h5_ds_names[:4])

        if self._vdl:
            h5_dss_list.append('dts_vars')

        if self._bs_flag:
            h5_dss_list.append('boot_arrs')

        assert h5_dss_list

        n_dss = len(h5_dss_list)

        for i in range(n_dss):
            dss = self._h5_hdl.create_group(h5_dss_list[i])
            var_labs = self.var_labs_list[i]

            for lab in var_labs:

                try:
                    var = getattr(self, lab)

                except AttributeError:
                    continue

                if isinstance(var, np.ndarray):
                    dss[lab] = var

                elif isinstance(var, (str, int)):
                    dss.attrs[lab] = var

                elif isinstance(var, Path):
                    dss.attrs[lab] = str(var)

                elif ((self._twt == 'month') or
                      (self._twt == 'year')) and (lab == '_t_idx'):

                    _td = pd.Timedelta('1s')
                    _min_t = pd.Timestamp("1970-01-01")

                    dss[lab] = (self._t_idx - _min_t) // _td

                else:
                    raise KeyError(lab, type(var))

        self._h5_hdl.flush()
        return

    def _aft_app_dis(self):

        if self._fh_flag == 0:
            self._ut_hdf5()
        return

    def _ut_hdf5(self):

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

        if refr.shape[0] and test.shape[0]:
            dts = dftn(refr, test, self._uvecs, self._n_cpus)

        else:
            dts = np.array([], dtype=np.int64)

        return dts

    def _fill_get_rat(self, refr, test, i, j, farr):

        test_dts = self._get_dts(refr, test)

        if test_dts.shape[0]:
            farr[i, j] = (test_dts == 0).sum() / test_dts.shape[0]

        return test_dts

    def _fill_get_rat_bs(self, refr, test, i, j, farr_ul, farr_ll, farr_flgs):

        test_dts = self._get_dts(refr, test)

        farr_flgs[i, j] = True

        if test_dts.shape[0]:
            rat = (test_dts == 0).sum() / test_dts.shape[0]

            farr_ul[i, j] = max(farr_ul[i, j], rat)
            farr_ll[i, j] = min(farr_ll[i, j], rat)

        return test_dts

    def _pld_upld_rats(self, refr, test, refr_pld, i, j, *args):

        '''Just to have less white space
        '''

        test_test_dts = self._get_dts(test, test)
        test_pldis = test_test_dts > self._pl_dth
        test_pld = test[test_pldis, :]

        self._fill_get_rat(refr_pld, test_pld, i, j, self._pld)

        if self._bs_flag:
            ris, tis, rpis, cd_arr = args[0], args[1], args[2], args[3]

            tpis = np.zeros_like(tis, dtype=bool)
            tpis = tis | tpis
            tpis[tis] = test_pldis

            self._cmpt_bs_lims(
                rpis,
                tpis,
                cd_arr,
                i,
                j,
                self._pld_bs_ul,
                self._pld_bs_ll,
                self._pld_bs_flg)

        if self._ans_stl == 'alt_peel':
            self._fill_get_rat(refr_pld, test, i, j, self._pld_upld)
            self._fill_get_rat(refr, test_pld, i, j, self._upld_pld)

            if self._bs_flag:
                self._cmpt_bs_lims(
                    rpis,
                    tis,
                    cd_arr,
                    i,
                    j,
                    self._pld_upld_bs_ul,
                    self._pld_upld_bs_ll,
                    self._pld_upld_bs_flg)

                self._cmpt_bs_lims(
                    ris,
                    tpis,
                    cd_arr,
                    i,
                    j,
                    self._upld_pld_bs_ul,
                    self._upld_pld_bs_ll,
                    self._upld_pld_bs_flg)

        return

    def _cmpt_bs_lims(
            self,
            ris,
            tis,
            cd_arr,
            i,
            j,
            farr_ul,
            farr_ll,
            farr_flgs):

        if farr_flgs[j, i]:
            farr_ul[i, j] = farr_ul[j, i]
            farr_ll[i, j] = farr_ll[j, i]
            farr_flgs[i, j] = farr_flgs[j, i]
            return

        rmwr = self._mwr[ris]
        tmwr = self._mwr[tis]
        rtmwrs = ris | tis

        n_refr = rmwr.shape[0]
#         n_test = tmwr.shape[0]

        n_rbsis = np.unique(rmwr).shape[0] + np.unique(tmwr).shape[0]

        unq_rtmwrs = np.unique(np.concatenate((rmwr, tmwr)))

        for _ in range(self._n_bs):
            rbsis = np.random.choice(
                unq_rtmwrs,
                size=n_rbsis,
                replace=True)

            bs_set = []
            for ibs in rbsis:
                ibsis = (self._mwr == ibs) & rtmwrs
                bs_set.append(cd_arr[ibsis, :].copy('c'))

            bs_set = np.concatenate(bs_set, axis=0)

#             if self.verbose:
#                 print('\t\tbsno:', _, bs_set.shape[0], (n_refr + n_test))

#             assert bs_set.shape[0] == (n_refr + n_test)
#             assert bs_set.shape[1] == self._ans_dims

            refr_bs = bs_set[:n_refr, :]
            test_bs = bs_set[n_refr:, :]

            self._fill_get_rat_bs(
                refr_bs, test_bs, i, j, farr_ul, farr_ll, farr_flgs)
        return

