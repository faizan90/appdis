'''
@author: Faizan-Uni-Stuttgart

'''

import os
import timeit
import time
import pickle
from pathlib import Path

from appdis import (
    AppearDisappearData,
    AppearDisappearSettings,
    AppearDisappearAnalysis,
    AppearDisappearPlot)


def main():
    main_dir = Path(r'P:\Synchronize\IWS\2016_DFG_SPATE\data\moving_window_volumes_test_01\ecad')
    os.chdir(main_dir)

    in_var_file = main_dir / r'ecad_pp_anomaly_pca_1961_2015.pkl'

    n_uvecs = int(1e3)
    n_cpus = 'auto'
    fig_size = (15, 14)
    n_dims = 6
    ws = 10 * 12  # window size
    analysis_style = 'raw'
    time_win_type = 'month'

    n_boots = 0

    out_dir = (f'anom_pca_{n_uvecs:1.0E}_uvecs_{n_dims}_dims_{ws}_ws_'
               f'{analysis_style}_as_{time_win_type}_twt_{n_boots}_bs')

    peel_depth = 0  # greater than this are kept

    print('out_dir:', out_dir)

    with open(in_var_file, 'rb') as _hdl:
        in_var_dict = pickle.load(_hdl)
        tot_in_var_arr = in_var_dict['pcs_arr'][:4100]
        time_idx = in_var_dict['anomaly_var_df'].index[:4100]
#         eig_val_cum_sums = in_var_dict['eig_val_cum_sums']

    ad_data = AppearDisappearData()
    ad_data.set_data_array(tot_in_var_arr)
    ad_data.set_time_index(time_idx)
    ad_data.generate_and_set_unit_vectors(n_dims, n_uvecs, n_cpus)
    ad_data.verify()

    ad_sett = AppearDisappearSettings()
    ad_sett.set_analysis_parameters(
        ws,
        time_win_type,
        analysis_style,
        n_dims,
        peel_depth,
        n_cpus)
    ad_sett.set_boot_strap_on_off(n_boots)
    ad_sett.set_outputs_directory(out_dir)
    ad_sett.save_outputs_to_hdf5_on_off(True)
    ad_sett.verify()

    ad_ans = AppearDisappearAnalysis()
    ad_ans.set_data(ad_data)
    ad_ans.set_settings(ad_sett)
    ad_ans.verify()

    ad_ans.cmpt_appear_disappear()
    ad_ans.close_hdf5()

#     hdf5_path = ad_ans._h5_path
    hdf5_path = Path(out_dir) / 'app_dis_ds.hdf5'

    ad_plot = AppearDisappearPlot()
    ad_plot.set_hdf5(hdf5_path)
    ad_plot.set_outputs_directory(out_dir)
    ad_plot.set_fig_size(fig_size)
    ad_plot.verify()

    ad_plot.plot_app_dis()

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
