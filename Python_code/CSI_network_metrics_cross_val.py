
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
from itertools import combinations, permutations


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('activities', help='Activities to be considered')
    parser.add_argument('n_tot', help='Number of streams * number of antennas', type=int)
    parser.add_argument('names_base', help='Names base for the files')
    parser.add_argument('num_folders', help='Number of folders', type=int)
    parser.add_argument('num_folders_train', help='Number of folders for training', type=int)
    parser.add_argument('num_folders_val', help='Number of folders for validation', type=int)
    parser.add_argument('--bandwidth', help='Bandwidth in [MHz] to select the subcarriers, can be 20, 40, 80 '
                                            '(default 80)', default=80, required=False, type=int)
    parser.add_argument('--sub_band', help='Sub_band idx in [1, 2, 3, 4] for 20 MHz, [1, 2] for 40 MHz '
                                           '(default 1)', default=1, required=False, type=int)
    parser.add_argument('--sub_sampling', help='Sampling in [1, 2, 3, 4, 5]'
                                               '(default 1)', default=1, required=False, type=int)
    args = parser.parse_args()

    n_antennas = args.n_tot
    num_folders = args.num_folders
    num_folders_train = args.num_folders_train
    num_folders_val = args.num_folders_val

    bandwidth = args.bandwidth
    sub_band = args.sub_band
    sub_sampling = args.sub_sampling

    csi_act = args.activities
    activities = []
    for lab_act in csi_act.split(','):
        activities.append(lab_act)
    activities = np.asarray(activities)
    num_act = activities.shape[0]

    suffix = '_bandw' + str(bandwidth) + '_RU' + str(sub_band) + '_sampling' + str(sub_sampling)

    folders_idx = list(np.arange(num_folders))
    num_elements_comb = 2
    list_sets_name = ['train', 'val', 'test']
    num_elements_permut = 2

    n_combinations = np.math.factorial(num_folders) / (np.math.factorial(num_elements_comb) * np.math.factorial(num_folders - num_elements_comb))
    n_permutations = np.math.factorial(num_elements_permut)
    n_comb_perm = int(n_combinations * n_permutations)

    names_string = args.names_base
    names_files = []
    for nam in names_string.split(','):
        names_files.append(nam)
    names_files = np.asarray(names_files)
    num_files = names_files.shape[0]

    n_tot_entries = n_comb_perm*num_files
    accuracies_cross_val = np.zeros((n_tot_entries, num_act))
    fscores_cross_val = np.zeros((n_tot_entries, num_act))
    avg_accuracies_cross_val_antennas = np.zeros((n_tot_entries, n_antennas))
    avg_fscores_cross_val_antennas = np.zeros((n_tot_entries, n_antennas))

    for idx_name, name_b in enumerate(names_files):

        index_comb_perm = 0
        comb_train = combinations(folders_idx, num_elements_comb)
        for train_set in comb_train:
            folders_idx_val_test = set(folders_idx).difference(train_set)
            perm_val_test = permutations(folders_idx_val_test, num_elements_permut)
            for val_test_indices in perm_val_test:
                val_indices = list(val_test_indices[:num_folders_val])
                test_indices = list(val_test_indices[num_folders_val:])
                train_indices = list(train_set)

                train_test_val_name = 'train_' + str(np.asarray(train_indices)+1) + '_val_' \
                                      + str(np.asarray(val_indices)+1)  + '_test_' + str(np.asarray(test_indices)+1)
                name_base = name_b + '_' + train_test_val_name + '_' + str(csi_act) + '_' + suffix

                name_file = './outputs/test_' + name_base + '.txt'

                try:
                    with open(name_file, "rb") as fp:  # Pickling
                        conf_matrix_dict = pickle.load(fp)
                except FileNotFoundError:
                    print(name_file, ' not found')
                    continue

                # MERGE ANTENNAS
                conf_matrix_max_merge = conf_matrix_dict['conf_matrix_max_merge']
                conf_matrix_max_merge_normaliz_row = conf_matrix_max_merge / \
                                                     np.sum(conf_matrix_max_merge, axis=1).reshape(-1, 1)
                accuracies_max_merge = np.diag(conf_matrix_max_merge_normaliz_row)
                accuracy_max_merge = conf_matrix_dict['accuracy_max_merge']
                precision_max_merge = conf_matrix_dict['precision_max_merge']
                recall_max_merge = conf_matrix_dict['recall_max_merge']
                fscore_max_merge = conf_matrix_dict['fscore_max_merge']
                average_max_merge_prec = np.mean(precision_max_merge)
                average_max_merge_rec = np.mean(recall_max_merge)
                average_max_merge_f = np.mean(fscore_max_merge)
                print('\n-- FINAL DECISION --')
                print('max-merge - average accuracy %f, average precision %f, average recall %f, average fscore %f'
                      % (accuracy_max_merge, average_max_merge_prec, average_max_merge_rec, average_max_merge_f))
                print('fscores - empty %f, sitting %f, walking %f, running %f'
                      % (fscore_max_merge[0], fscore_max_merge[1], fscore_max_merge[2], fscore_max_merge[3]))
                print('accuracies - empty %f, sitting %f, walking %f, running %f'
                      % (accuracies_max_merge[0], accuracies_max_merge[1], accuracies_max_merge[2], accuracies_max_merge[3]))

                accuracies_cross_val[idx_name*n_comb_perm + index_comb_perm, :] = accuracies_max_merge
                fscores_cross_val[idx_name*n_comb_perm + index_comb_perm, :] = fscore_max_merge

                # CHANGING THE NUMBER OF MONITOR ANTENNAS
                name_file = './outputs/change_number_antennas_test_' + name_base + '.txt'
                with open(name_file, "rb") as fp:  # Pickling
                    metrics_matrix_dict = pickle.load(fp)

                average_accuracy_change_num_ant = metrics_matrix_dict['average_accuracy_change_num_ant']
                average_fscore_change_num_ant = metrics_matrix_dict['average_fscore_change_num_ant']
                print('\naccuracies - one antenna %f, two antennas %f, three antennas %f, four antennas %f'
                      % (average_accuracy_change_num_ant[0], average_accuracy_change_num_ant[1], average_accuracy_change_num_ant[2],
                         average_accuracy_change_num_ant[3]))
                print('fscores - one antenna %f, two antennas %f, three antennas %f, four antennas %f'
                      % (average_fscore_change_num_ant[0], average_fscore_change_num_ant[1], average_fscore_change_num_ant[2],
                         average_fscore_change_num_ant[3]))

                avg_accuracies_cross_val_antennas[idx_name*n_comb_perm + index_comb_perm, :] = average_accuracy_change_num_ant
                avg_fscores_cross_val_antennas[idx_name*n_comb_perm + index_comb_perm, :] = average_fscore_change_num_ant

                index_comb_perm += 1

    avg_accuracies_cross_val = np.mean(accuracies_cross_val, axis=0)
    avg_accuracy_cross_val = np.mean(avg_accuracies_cross_val)

    avg_fscores_cross_val = np.mean(fscores_cross_val, axis=0)
    avg_fscore_cross_val = np.mean(avg_fscores_cross_val)

    metrics_matrix_dict = {'accuracies_cross_val': accuracies_cross_val,
                           'avg_accuracies_cross_val': avg_accuracies_cross_val,
                           'fscores_cross_val': fscores_cross_val,
                           'avg_fscores_cross_val': avg_fscores_cross_val
                           }

    name_file_save = './evaluations/' + args.names_base + '_' + str(csi_act) + '_' + suffix + '.txt'
    with open(name_file_save, "wb") as fp:  # Pickling
        pickle.dump(metrics_matrix_dict, fp)
