
"""
    Copyright (C) 2023 Francesca Meneghello
    contact: meneghello@dei.unipd.it
    This program is free software: you can redistribute it and/or modify
    it under the terms of the GNU General Public License as published by
    the Free Software Foundation, either version 3 of the License, or
    (at your option) any later version.
    This program is distributed in the hope that it will be useful,
    but WITHOUT ANY WARRANTY; without even the implied warranty of
    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
    GNU General Public License for more details.
    You should have received a copy of the GNU General Public License
    along with this program.  If not, see <https://www.gnu.org/licenses/>.
"""

import argparse
import numpy as np
import pickle
import math as mt
import os
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib import rcParams

rcParams['font.family'] = 'serif'
rcParams['font.serif'] = 'Times'
rcParams['text.usetex'] = 'true'
rcParams['text.latex.preamble'] = [r'\usepackage{newtxmath}']
rcParams['font.size'] = 16


def plt_fft_doppler_antennas(doppler_spectrum_list, sliding_lenght, delta_v, name_plot):
    if doppler_spectrum_list:
        fig = plt.figure()
        gs = gridspec.GridSpec(4, 1, figure=fig)
        step = 15
        length_v = mt.floor(doppler_spectrum_list[0].shape[1] / 2)
        factor_v = step * (mt.floor(length_v / step))
        ticks_y = np.arange(length_v - factor_v, length_v + factor_v + 1, step)
        ticks_x = np.arange(0, doppler_spectrum_list[0].shape[0], int(doppler_spectrum_list[0].shape[0]/20))
        ax = []

        for p_i in range(len(doppler_spectrum_list)):
            ax1 = fig.add_subplot(gs[(p_i, 0)])
            plt1 = ax1.pcolormesh(doppler_spectrum_list[p_i].T, cmap='viridis', linewidth=0, rasterized=True)
            plt1.set_edgecolor('face')
            cbar1 = fig.colorbar(plt1)
            cbar1.ax.set_ylabel('power [dB]', rotation=270, labelpad=14)
            ax1.set_ylabel(r'velocity [m/s]')
            ax1.set_xlabel(r'time [s]')
            ax1.set_yticks(ticks_y + 0.5)
            ax1.set_yticklabels(np.round((ticks_y - length_v) * delta_v, 2))
            ax1.set_xticks(ticks_x)
            ax1.set_xticklabels(np.round(ticks_x * sliding_lenght * 6e-3, 2))
            ax.append(ax1)

        for axi in ax:
            axi.label_outer()
        fig.set_size_inches(20, 10)
        plt.savefig(name_plot, bbox_inches='tight')
        plt.close()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('dir_doppler', help='Directory of data')
    parser.add_argument('subdirs', help='Sub directory of data')
    parser.add_argument('sample_length', help='Number of packet in a window', type=int)
    parser.add_argument('sliding', help='Number of packet for sliding operations', type=int)
    parser.add_argument('end_plt', help='End index to plot', type=int)
    parser.add_argument('noise_level', help='Level for the noise to be removed (pay attention that noise has been '
                                            'removed also when computing Doppler, here you can only remove more noise)',
                        type=float)
    parser.add_argument('--bandwidth', help='Bandwidth in [MHz] to select the subcarriers, can be 20, 40, 80 '
                                            '(default 80)', default=80, required=False, type=int)
    parser.add_argument('--sub_band', help='Sub_band idx in [1, 2, 3, 4] for 20 MHz, [1, 2] for 40 MHz '
                                           '(default 1)', default=1, required=False, type=int)
    parser.add_argument('--sub_sampling', help='Sampling in [1, 2, 3, 4, 5]'
                                               '(default 1)', default=1, required=False, type=int)

    args = parser.parse_args()

    bandwidth = args.bandwidth
    sub_band = args.sub_band
    sub_sampling = args.sub_sampling
    noise_lev = args.noise_level

    num_symbols = args.sample_length
    middle = int(mt.floor(num_symbols / 2))

    n_tot = 4
    Tc = 7.5e-3
    fc = 5785e6
    v_light = 3e8
    delta_v = round(v_light / (Tc * fc * num_symbols), 3)

    sliding = args.sliding
    list_subdir = args.subdirs

    for subdir in list_subdir.split(','):
        path_doppler = args.dir_doppler + subdir

        activity = subdir[0]

        csi_d_antennas = []
        for i_ant in range(n_tot):
            path_doppler_name = path_doppler + '/' + subdir + '_stream_' + str(i_ant) + '_bandw' + str(bandwidth) + \
                        '_RU' + str(sub_band) + '_sampling' + str(sub_sampling) + '.txt'

            print(path_doppler_name)

            with open(path_doppler_name, "rb") as fp:  # Pickling
                csi_d_profile_array = pickle.load(fp)
            csi_d_profile_array[csi_d_profile_array < mt.pow(10, noise_lev)] = mt.pow(10, noise_lev)

            csi_d_profile_array_log = 10 * np.log10(csi_d_profile_array)
            middle = int(np.floor(csi_d_profile_array_log.shape[1] / 2))

            csi_d_profile_array_log = csi_d_profile_array_log[:min(csi_d_profile_array_log.shape[0], args.end_plt), :]

            csi_d_antennas.append(csi_d_profile_array_log)

        name_p = './plots/csi_doppler_activity_' + subdir + '_' + activity + '_bandw' + str(bandwidth) + \
                 '_RU' + str(sub_band) + '_sampling' + str(sub_sampling) + '.png'

        plt_fft_doppler_antennas(csi_d_antennas, sliding, delta_v, name_p)
