'''
@author: Faizan-Uni-Stuttgart

'''

import os
import timeit
import time
import pickle
from pathlib import Path

from appdis import (
    AppearDisappearAnalysis,
    AppearDisappearPlot)


def main():
    main_dir = Path(r'P:\Synchronize\IWS\2016_DFG_SPATE\data\moving_window_volumes_test_01\ecad')
    os.chdir(main_dir)

    in_var_file = main_dir / r'ecad_pp_anomaly_pca_1961_2015.pkl'

    n_uvecs = int(1e2)
    n_cpus = 'auto'
    n_dims = 3
    ws = 5  # window size
    analysis_style = 'peel'
    time_win_type = 'year'
    n_ticks = 20
    cmap = 'jet'

    peel_depth = 1  # greater than this are kept
    n_boots = 0
    hdf_flush_flag = 0
    vol_data_lev = 1
    loo_flag = False

    ecop_style = 'un_peel'
    ecop_data_type = 'window'

    ann_flag = False
    plot_flag = False
#     ann_flag = True
#     plot_flag = True

    out_dir = (f'anom_pca_{n_uvecs:1.0E}_uvecs_{n_dims}_dims_{ws}_ws_'
               f'{analysis_style}_as_{time_win_type}_twt_{n_boots}_bs_'
               f'{peel_depth}_pldt_2')

    print('out_dir:', out_dir)

    hdf5_path = Path(out_dir) / 'app_dis_ds.hdf5'

    with open(in_var_file, 'rb') as _hdl:
        in_var_dict = pickle.load(_hdl)
        tot_in_var_arr = in_var_dict['pcs_arr']  # [:4100]
        time_idx = in_var_dict['anomaly_var_df'].index  # [:4100]
#         eig_val_cum_sums = in_var_dict['eig_val_cum_sums']

    if ann_flag:
        ad_ans = AppearDisappearAnalysis()
        ad_ans.set_data_array(tot_in_var_arr)
        ad_ans.set_time_index(time_idx)
        ad_ans.generate_and_set_unit_vectors(n_dims, n_uvecs, n_cpus)

        ad_ans.set_analysis_parameters(
            time_win_type,
            ws,
            analysis_style,
            n_dims,
            peel_depth,
            n_cpus)

        ad_ans.set_boot_strap_on_off(n_boots)
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
        ad_plot.set_fig_props(n_ticks, cmap)
        ad_plot.verify()
        ad_plot.set_n_cpus(n_cpus)  # must call after verify to take effect

        ad_plot.plot_app_dis()
        ad_plot.plot_volumes(loo_flag)
        ad_plot.plot_ecops(ecop_style, ecop_data_type)
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
