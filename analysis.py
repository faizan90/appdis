'''
@author: Faizan-Uni-Stuttgart

'''

import h5py
import numpy as np
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

        self._data_set_flag = False
        self._sett_set_flag = False
        self._in_vrfd_flag = False
        self._mw_rng_cmptd_flag = False
        return

    def set_data(self, appear_disappear_data_obj):

        assert isinstance(appear_disappear_data_obj, AppearDisappearData)
        assert appear_disappear_data_obj._in_vrfd_flag

        addo = appear_disappear_data_obj

        vars_list = [
            '_data_arr',
            '_t_idx',
            '_t_idx_t',
            '_uvecs',
            '_n_data_pts',
            '_n_data_dims',
            '_n_uvecs',
            ]

        for v in vars_list:
            setattr(self, v, getattr(addo, v))

        self._data_set_flag = True
        return

    def set_settings(self, appear_disappear_settings_obj):

        assert isinstance(
            appear_disappear_settings_obj, AppearDisappearSettings)
        assert appear_disappear_settings_obj._in_vrfd_flag

        adso = appear_disappear_settings_obj

        vars_list = [
            '_ws',
            '_nms',
            '_twt',
            '_ans_stl',
            '_ans_dims',
            '_pl_dth',
            '_n_cpus',
            '_out_dir',
            '_bs_flag',
            '_n_bs',
            '_hdf5_flag',
            ]

        for v in vars_list:
            setattr(self, v, getattr(adso, v))

        self._sett_set_flag = True
        return

    def verify(self):

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
        assert (self._ws + self._nms) < self._n_data_pts

        self._in_vrfd_flag = True
        return

    def cmpt_appear_disappear(self):

        self._bef_app_dis()

        pl_flg = (self._ans_stl == 'peel') or (self._ans_stl == 'alt_peel')

        cd_arr = self._data_arr[:, :self._ans_dims]

        # computations for _ans_stl == 'raw' are kept here. The rest are
        # passed to other functions to avoid a long loop and too much white
        # space

        for i in range(0, self._mwi, self._nms):
            ris = (self._mwr >= i) & (self._mwr < (i + self._ws))
            if not ris.sum():
                continue

            if self.verbose:
                print('\n')
                _ridx = np.where(ris)[0]
                rbeg_time = self._t_idx[_ridx[0]]
                rend_time = self._t_idx[_ridx[-1]]
                print('Reference begin and end time:', rbeg_time, rend_time)

            refr = cd_arr[ris, :].copy('c')

            if pl_flg:
                refr_refr_dts = self._get_dts(refr, refr)
                refr_pldis = refr_refr_dts > self._pl_dth
                refr_pld = refr[refr_pldis, :]

                if self._bs_flag:
                    rpis = np.zeros_like(ris, dtype=bool)
                    rpis = ris | rpis
                    rpis[ris] = refr_pldis

            for j in range(self._mwi):
                if i == j:
                    continue

                tis = (self._mwr >= j) & (self._mwr < (j + self._ws))
                if not tis.sum():
                    continue

                if self.verbose:
                    _tidx = np.where(tis)[0]
                    tbeg_time = self._t_idx[_tidx[0]]
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

        self._aft_app_dis()
        return

    def close_hdf5(self):

        if self._hdf5_flag:
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

        self._mwr = win_rng
        self._mwi = np.arange(0, mwi, self._nms).shape[0] + 1

        assert self._mwi > 1
        assert self._mwi > self._nms

        self._mw_rng_cmptd_flag = True
        return

    def _bef_app_dis(self):

        assert self._in_vrfd_flag

        self._cmpt_mw_rng()
        assert self._mw_rng_cmptd_flag

        # ratio of test points outside refr
        self._upld = np.full(
            (self._mwi, self._mwi), np.nan, dtype=np.float64, order='c')

        if self._bs_flag:
            self._upld_bs_ul = self._upld.copy()
            self._upld_bs_ll = self._upld.copy()

            self._upld_bs_flg = np.zeros_like(
                self._upld, dtype=bool, order='c')

        if (self._ans_stl == 'peel') or (self._ans_stl == 'alt_peel'):

            self._pld = self._upld.copy()

            if self._bs_flag:
                self._pld_bs_ul = self._pld.copy()
                self._pld_bs_ll = self._pld.copy()

                self._pld_bs_flg = np.zeros_like(
                    self._pld, dtype=bool, order='c')

            if self._ans_stl == 'alt_peel':
                # in the name, first position is for refr, the second for test
                self._pld_upld = self._pld.copy()
                self._upld_pld = self._pld_upld.copy()

                if self._bs_flag:
                    self._pld_upld_bs_ul = self._pld.copy()
                    self._pld_upld_bs_ll = self._pld.copy()

                    self._upld_pld_bs_ul = self._pld_upld.copy()
                    self._upld_pld_bs_ll = self._pld_upld.copy()

                    self._pld_upld_bs_flg = np.zeros_like(
                        self._pld_upld, dtype=bool, order='c')

                    self._upld_pld_bs_flg = np.zeros_like(
                        self._upld_pld, dtype=bool, order='c')

        if self._hdf5_flag:
            h5_name = str(self._out_dir / 'app_dis_ds.hdf5')
            self._h5_hdl = h5py.File(h5_name, mode='w', driver='core')

            dg = self._h5_hdl.create_group('in_data')
            dg['data_arr'] = self._data_arr
            # dg['t_idx'] = self._t_idx.values
            dg.attrs['t_idx'] = self._t_idx_t
            dg['uvecs'] = self._uvecs
            dg.attrs['n_data_pts'] = self._n_data_pts
            dg.attrs['n_data_dims'] = self._n_data_dims
            dg.attrs['n_uvecs'] = self._n_uvecs

            ds = self._h5_hdl.create_group('settings')
            ds.attrs['ws'] = self._ws
            ds.attrs['nms'] = self._nms
            ds.attrs['twt'] = self._twt
            ds.attrs['ans_stl'] = self._ans_stl
            ds.attrs['ans_dims'] = self._ans_dims
            ds.attrs['pl_dth'] = self._pl_dth
            ds.attrs['n_cpus'] = self._n_cpus
#             ds.attrs['out_dir'] = self._out_dir
            ds.attrs['bs_flag'] = self._bs_flag
            ds.attrs['n_bs'] = self._n_bs

            ivs = self._h5_hdl.create_group('inter_vars')
            ivs['mwr'] = self._mwr
            ivs.attrs['mwi'] = self._mwi

            self._h5_hdl.flush()
        return

    def _aft_app_dis(self):

        if self._hdf5_flag:
            rds = self._h5_hdl.create_group('app_dis_arrs')

            rds['upld'] = self._upld

            if (self._ans_stl == 'peel') or (self._ans_stl == 'alt_peel'):
                rds['pld'] = self._pld

                if self._ans_stl == 'alt_peel':
                    rds['pld_upld'] = self._pld_upld
                    rds['upld_pld'] = self._upld_pld

            if self._bs_flag:
                bsds = self._h5_hdl.create_group('boot_arrs')

                bsds['upld_ul'] = self._upld_bs_ul
                bsds['upld_ll'] = self._upld_bs_ll
                bsds['upld_flg'] = self._upld_bs_flg

                if (self._ans_stl == 'peel') or (self._ans_stl == 'alt_peel'):

                    bsds['pld_ul'] = self._pld_bs_ul
                    bsds['pld_ll'] = self._pld_bs_ll
                    bsds['pld_flg'] = self._pld_bs_flg

                    if self._ans_stl == 'alt_peel':
                        bsds['pld_upld_ul'] = self._pld_upld_bs_ul
                        bsds['pld_upld_ll'] = self._pld_upld_bs_ll
                        bsds['pld_upld_flg'] = self._pld_upld_bs_flg

                        bsds['upld_pld_ul'] = self._upld_pld_bs_ul
                        bsds['upld_pld_ll'] = self._upld_pld_bs_ll
                        bsds['upld_pld_flg'] = self._upld_pld_bs_flg

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
        n_test = tmwr.shape[0]

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

            print('\t\t', bs_set.shape[0], (n_refr + n_test))
#             assert bs_set.shape[0] == (n_refr + n_test)
            assert bs_set.shape[1] == self._ans_dims

            refr_bs = bs_set[:n_refr, :]
            test_bs = bs_set[n_refr:, :]

            self._fill_get_rat_bs(
                refr_bs, test_bs, i, j, farr_ul, farr_ll, farr_flgs)
        return

