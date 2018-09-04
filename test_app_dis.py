'''
@author: Faizan-Uni-Stuttgart

'''

import os
import timeit
import time
import pickle
from pathlib import Path

import numpy as np
import pandas as pd

from appdis import (
    AppearDisappearAnalysis,
    AppearDisappearPlot)

np.set_printoptions(
    precision=3,
    threshold=2000,
    linewidth=200000,
    formatter={'float': '{:+0.3f}'.format})

pd.options.display.precision = 3
pd.options.display.max_rows = 100
pd.options.display.max_columns = 100
pd.options.display.width = 250


def main():

    main_dir = Path(r'P:\Synchronize\IWS\2016_DFG_SPATE\data\moving_window_volumes_test_01\ecad_pp')
    os.chdir(main_dir)

    in_var_file = main_dir / r'ecad_pp_anomaly_pca_1961_2015.pkl'

    n_uvecs = int(1e5)
    n_cpus = 7  # 'auto'
    n_dims = 6
    ws = 10  # window size
    analysis_style = 'un_peel'
    time_win_type = 'year'
    n_ticks = 20
    cmap = 'jet'
    steps = (365 * 12)

    peel_depth = 0  # greater than this are kept
    n_boots = 0
    nv_boots = 0
    hdf_flush_flag = 0
    vol_data_lev = 0
    loo_flag = False
    max_allowed_corr = 0.5
    app_dis_cb_max = 10

    sel_idxs_flag = False
    take_rest_flag = False
    ann_flag = False
    plot_flag = False

#     sel_idxs_flag = True
#     take_rest_flag = True
    ann_flag = True
    plot_flag = True

    if sel_idxs_flag:
        sel_idxs_lab = '_sel_idxs'
    else:
        sel_idxs_lab = ''

    if take_rest_flag:
        rest_lab = '_rest'
    else:
        rest_lab = ''

    out_dir = (f'anom_pca_{n_uvecs:1.0E}_uvecs_{n_dims}_dims_{ws}_ws_'
               f'{analysis_style}_as_{time_win_type}_twt_{n_boots}_bs_'
               f'{peel_depth}_pldt_{nv_boots}_vbs{sel_idxs_lab}{rest_lab}_new_dts')

    print('out_dir:', out_dir)

    hdf5_path = Path(out_dir) / 'app_dis_ds.hdf5'

    if ann_flag:
        with open(in_var_file, 'rb') as _hdl:
            in_var_dict = pickle.load(_hdl)
            in_anom_df = in_var_dict['anomaly_var_df'].iloc[:steps]

            if sel_idxs_flag:
                tot_in_var_arr = in_anom_df.values.copy('c')

            else:
                tot_in_var_arr = in_var_dict['pcs_arr'][:steps, :].copy('c')

                if take_rest_flag:
                    rest_arr = tot_in_var_arr[:, n_dims - 1:]
                    rest_arr = (rest_arr ** 2).sum(axis=1) ** 0.5
                    rest_arr = rest_arr.reshape(-1, 1)

                    tot_in_var_arr = np.hstack(
                        (tot_in_var_arr[:, :n_dims - 1], rest_arr))

                    assert tot_in_var_arr.shape[1] == n_dims

            time_idx = in_anom_df.index
            del in_var_dict, in_anom_df

        ad_ans = AppearDisappearAnalysis()
        ad_ans.set_data_array(tot_in_var_arr)
        ad_ans.set_time_index(time_idx)
        ad_ans.generate_and_set_unit_vectors(n_dims, n_uvecs, n_cpus)

        ad_ans.set_analysis_parameters(
            time_win_type,
            ws,
            analysis_style,
            peel_depth,
            n_cpus)

        if sel_idxs_flag:
            ad_ans.set_optimization_parameters(
                0.5,
                0.95,
                150,
                20000,
                5000,
                max_allowed_corr)

        ad_ans.set_boot_strap_on_off(n_boots)
        ad_ans.set_volume_boot_strap_on_off(nv_boots)
        ad_ans.set_outputs_directory(out_dir)
        ad_ans.save_outputs_to_hdf5_on_off(True, hdf_flush_flag)

        ad_ans.save_volume_data_level(vol_data_lev)

        ad_ans.verify()

#         ad_ans.resume_from_hdf5(hdf5_path)

        ad_ans.cmpt_appear_disappear()
        ad_ans.terminate_analysis()

    if plot_flag:
        ad_plot = AppearDisappearPlot()
        ad_plot.set_hdf5(hdf5_path)
        ad_plot.set_outputs_directory(out_dir)
        ad_plot.set_fig_props(n_ticks, cmap, app_dis_cb_max)
        ad_plot.verify()

        ad_plot.plot_app_dis()

        if sel_idxs_flag:
            ad_plot.plot_sim_anneal_opt()

        if vol_data_lev:
            ad_plot.plot_volumes(loo_flag)

            ad_plot.plot_ecops()
    return


if __name__ == '__main__':
    _save_log_ = False
    if _save_log_:
        from datetime import datetime
        from std_logger import StdFileLoggerCtrl

        # save all console activity to out_log_file
        out_log_file = os.path.join(r'P:\\',
                                    r'Synchronize',
                                    r'python_script_logs',
                                    ('%s_log_%s.log' % (
                                    os.path.basename(__file__),
                                    datetime.now().strftime('%Y%m%d%H%M%S'))))
        log_link = StdFileLoggerCtrl(out_log_file)

    print('#### Started on %s ####\n' % time.asctime())
    START = timeit.default_timer()  # to get the runtime of the program

    main()

    STOP = timeit.default_timer()  # Ending time
    print(('\n#### Done with everything on %s.\nTotal run time was'
           ' about %0.4f seconds ####' % (time.asctime(), STOP - START)))

    if _save_log_:
        log_link.stop()
