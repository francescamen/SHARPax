
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
import scipy.io as sio
from os import listdir
import pickle
from os import path


def hampel_filter(input_matrix, window_size, n_sigmas=3):
    n = input_matrix.shape[1]
    new_matrix = np.zeros_like(input_matrix)
    k = 1.4826  # scale factor for Gaussian distribution

    for ti in range(n):
        start_time = max(0, ti - window_size)
        end_time = min(n, ti + window_size)
        x0 = np.nanmedian(input_matrix[:, start_time:end_time], axis=1, keepdims=True)
        s0 = k * np.nanmedian(np.abs(input_matrix[:, start_time:end_time] - x0), axis=1)
        mask = (np.abs(input_matrix[:, ti] - x0[:, 0]) > n_sigmas * s0)
        new_matrix[:, ti] = mask*x0[:, 0] + (1 - mask)*input_matrix[:, ti]

    return new_matrix


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('dir', help='Directory of data')
    parser.add_argument('all_dir', help='All the files in the directory, default no', type=int, default=0)
    parser.add_argument('name', help='Name of experiment file')
    parser.add_argument('nss', help='Number of spatial streams', type=int)
    parser.add_argument('ncore', help='Number of cores', type=int)
    parser.add_argument('nsubchannels', help='Number of subchannels', type=int)
    parser.add_argument('start_idx', help='Idx where start processing for each stream', type=int)
    args = parser.parse_args()

    exp_dir = args.dir
    names = []

    if args.all_dir:
        all_files = listdir(exp_dir)
        mat_files = []
        for i in range(len(all_files)):
            if all_files[i].endswith('.mat'):
                names.append(all_files[i][:-4])
    else:
        names.append(args.name)

    for name in names:
        name_file = './phase_processing/signal_' + name + '.txt'
        if path.exists(name_file):
            print('Already processed')
            continue

        csi_buff_file = exp_dir + name + ".mat"
        csi_buff_struct = sio.loadmat(csi_buff_file)
        csi_buff_struct = (csi_buff_struct['cores'])

        npkt = csi_buff_struct.shape[1]
        ncore = args.ncore
        nsubchannels = args.nsubchannels

        csi_buff = np.zeros((nsubchannels, npkt, ncore), dtype=complex)

        matrix_idx = 0
        for pkt_idx in range(npkt):
            for core_idx in range(ncore):
                inserted = True
                try:
                    csi_buff[:, matrix_idx, core_idx] = csi_buff_struct[0][pkt_idx][0, core_idx]['nss'][0][0][0, 0]['data'][0][0]
                except IndexError:
                    inserted = False
                    continue
            if inserted:
                matrix_idx += 1

        csi_buff = csi_buff[:, :matrix_idx - 1, :]

        csi_buff = np.fft.fftshift(csi_buff, axes=0)

        delete_idxs = np.argwhere(np.sum(np.sum(csi_buff, axis=0), axis=1) == 0)[:, 0]  # packets empty
        csi_buff = np.delete(csi_buff, delete_idxs, axis=1)

        delete_idxs = np.asarray([-512, -511, -510, -509, -508, -507, -506, -505, -504, -503, -502, -501,
                                  -2, -1, 0, 1, 2,
                                  501, 502, 503, 504, 505, 506, 507, 508, 509, 510, 511], dtype=int) + 512
        pilot_subcarriers = np.asarray([-468, -400, -334, -266, -158, -92, -24, 24, 92, 158, 226, 334, 400, 468]) + 512
        csi_buff = np.delete(csi_buff, delete_idxs, axis=0)

        n_ss = args.nss
        n_core = args.ncore
        n_tot = n_ss * n_core

        start = args.start_idx  # 1000
        end = csi_buff.shape[1]
        signal_complete = csi_buff[:, start:end, :]

        name_file = './phase_processing/signal_' + name + '.txt'
        with open(name_file, "wb") as fp:  # Pickling
            pickle.dump(signal_complete, fp)
