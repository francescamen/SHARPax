
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
import glob
import os
import numpy as np
import pickle
import math as mt
import shutil
from dataset_utility import create_windows_antennas, convert_to_number
from itertools import combinations, permutations


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('dir', help='Directory of data')
    parser.add_argument('num_folders', help='Number of folders', type=int)
    parser.add_argument('num_folders_train', help='Number of folders for training', type=int)
    parser.add_argument('num_folders_val', help='Number of folders for validation', type=int)
    parser.add_argument('window_length', help='Number of samples per window', type=int)
    parser.add_argument('stride_length', help='Number of samples to stride', type=int)
    parser.add_argument('labels_activities', help='Labels of the activities to be considered')
    parser.add_argument('n_tot', help='Number of streams * number of antennas', type=int)
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
    suffix = '_bandw' + str(bandwidth) + '_RU' + str(sub_band) + '_sampling' + str(sub_sampling)

    labels_activities = args.labels_activities
    csi_label_dict = []
    for lab_act in labels_activities.split(','):
        csi_label_dict.append(lab_act)
    activities = np.asarray(labels_activities)

    n_tot = args.n_tot
    exp_dir = args.dir
    save_dir = exp_dir + 'dataset_train_val_test/'
    if not os.path.exists(save_dir):
        os.mkdir(save_dir)
    path_save = save_dir + str(activities) + '_bandw' + str(bandwidth) + '_RU' + str(sub_band) + \
                                '_sampling' + str(sub_sampling)
    if os.path.exists(path_save):
        remove_files = glob.glob(path_save + '/*')
        for f in remove_files:
            shutil.rmtree(f)
    else:
        os.mkdir(path_save)

    window_length = args.window_length  # number of windows considered
    stride_length = args.stride_length

    num_folders = args.num_folders
    num_folders_train = args.num_folders_train
    num_folders_val = args.num_folders_val

    names_files_folders = []
    labels_wind_files_folders = []
    num_wind_files_folders = []
    for folder_idx in range(1, num_folders + 1):
        save_dir_folder = save_dir + str(activities) + suffix + '/' + str(folder_idx) + '/'
        if not os.path.exists(save_dir_folder):
            os.mkdir(save_dir_folder)
        csi_matrices = []
        labels_tot = []
        lengths_tot = []
        for act in csi_label_dict:
            name = act + str(folder_idx)
            if act != 'E':
                name = name + '_P1'
                # E1, E2, E3, E4, W1_P1, W2_P1, W3_P1, W4_P1, R1_P1, R2_P1, R3_P1, R4_P1, S1_P1, S2_P1, S3_P1, S4_P1
            csi_matrix = []
            label = convert_to_number(act, csi_label_dict)
            for i_ant in range(n_tot):
                name_file = exp_dir + name + '/' + name + '_stream_' + str(i_ant) + '_bandw' + str(bandwidth) + \
                            '_RU' + str(sub_band) + '_sampling' + str(sub_sampling) + '.txt'
                with open(name_file, "rb") as fp:  # Unpickling
                    stft_sum_1 = pickle.load(fp)
                stft_sum_1[stft_sum_1 < mt.pow(10, noise_lev)] = mt.pow(10, noise_lev)
                stft_sum_1_mean = stft_sum_1 - np.mean(stft_sum_1, axis=0, keepdims=True)
                csi_matrix.append(stft_sum_1_mean.T)
            lengths_tot.append(stft_sum_1_mean.shape[0])
            labels_tot.append(label)
            csi_matrices.append(np.asarray(csi_matrix))

        csi_matrices_wind, labels_wind = create_windows_antennas(csi_matrices, labels_tot,
                                                               window_length, stride_length, remove_mean=False)
        num_windows = np.floor((np.asarray(lengths_tot) - window_length - 1) / stride_length + 1)
        if not len(csi_matrices_wind) == np.sum(num_windows):
            print('ERROR - shapes mismatch', len(csi_matrices_wind), np.sum(num_windows))

        names_files = []
        for ii in range(len(csi_matrices_wind)):
            name_file = save_dir_folder + str(ii) + '.txt'
            names_files.append(name_file)
            with open(name_file, "wb") as fp:  # Pickling
                pickle.dump(csi_matrices_wind[ii], fp)
        names_files_folders.append(names_files)
        labels_wind_files_folders.append(labels_wind)
        num_wind_files_folders.append(num_windows)
        a = 1

    folders_idx = list(np.arange(num_folders))
    num_elements_comb = 2
    num_elements_permut = 2
    comb_train = combinations(folders_idx, num_elements_comb)
    list_sets_name = ['train', 'val', 'test']
    for train_set in comb_train:
        folders_idx_val_test = set(folders_idx).difference(train_set)
        perm_val_test = permutations(folders_idx_val_test, num_elements_permut)
        for val_test_indices in perm_val_test:
            val_indices = list(val_test_indices[:num_folders_val])
            test_indices = list(val_test_indices[num_folders_val:])
            train_indices = list(train_set)
            list_indices_sets = [train_indices, val_indices, test_indices]

            save_dir_folder = save_dir + str(activities) + suffix + '/train_' + str(np.asarray(train_indices)+1) + '_val_' \
                              + str(np.asarray(val_indices)+1)  + '_test_' + str(np.asarray(test_indices)+1) + '/'
            if not os.path.exists(save_dir_folder):
                os.mkdir(save_dir_folder)

            for set_idx in range(3):
                files_indices = list_indices_sets[set_idx]
                labels_files = []
                names_files_save = []
                num_wind_files = []
                for files_idx in files_indices:
                    labels_files.extend(labels_wind_files_folders[files_idx])
                    names_files_save.extend(names_files_folders[files_idx])
                    num_wind_files.extend(num_wind_files_folders[files_idx])

                name_labels = save_dir_folder + '/labels_' + list_sets_name[set_idx] + '_' + str(activities) + '.txt'
                with open(name_labels, "wb") as fp:  # Pickling
                    pickle.dump(labels_files, fp)
                name_f = save_dir_folder + '/files_' + list_sets_name[set_idx] + '_' + str(activities) + '.txt'
                with open(name_f, "wb") as fp:  # Pickling
                    pickle.dump(names_files_save, fp)
                name_f = save_dir_folder + '/num_windows_' + list_sets_name[set_idx] + '_' + str(activities) + '.txt'
                with open(name_f, "wb") as fp:  # Pickling
                    pickle.dump(num_wind_files, fp)
