
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
import matplotlib.pyplot as plt
import matplotlib as mpl
import matplotlib.cbook as cbook
from matplotlib.patches import Polygon
import scipy.stats as st
import scipy.optimize as so
from matplotlib.ticker import MultipleLocator
from matplotlib.ticker import FuncFormatter
from matplotlib.lines import Line2D

mpl.rcParams['font.family'] = 'serif'
mpl.rcParams['font.serif'] = 'Palatino'
mpl.rcParams['text.usetex'] = 'true'
mpl.rcParams['text.latex.preamble'] = [r'\usepackage{newtxmath}']
mpl.rcParams['font.size'] = 14
mpl.rcParams['axes.prop_cycle'] = plt.cycler(color=plt.cm.Accent.colors)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('activities', help='Activities to be considered')
    parser.add_argument('names_base', help='Names base for the files')
    args = parser.parse_args()

    names_string = args.names_base
    names_files = []
    for nam in names_string.split(','):
        names_files.append(nam)
    names_files = np.asarray(names_files)
    num_files = names_files.shape[0]

    csi_act = args.activities
    activities = []
    for lab_act in csi_act.split(','):
        activities.append(lab_act)
    activities = np.asarray(activities)
    num_act = activities.shape[0]

    #################################
    # BOX PLOT DIFFERENT BANDWIDTHS
    #################################
    n_entries = 7
    band_subband = [[80, 1], [40, 1], [40, 2], [20, 1], [20, 2], [20, 3], [20, 4]]
    band_subband_names = [r'RU1-996', r'RU1-484', r'RU2-484', r'RU1-242', r'RU2-242', r'RU3-242', r'RU4-242']

    avg_accuracies_cross_val = np.zeros((n_entries, num_act))
    avg_fscores_cross_val = np.zeros((n_entries, num_act))

    num_cross_val = 12
    avg_accuracies_activities = np.zeros((n_entries, num_cross_val*num_files))
    avg_fscores_activities = np.zeros((n_entries, num_cross_val*num_files))

    for idx, entry in enumerate(band_subband):
        bandwidth = entry[0]
        sub_band = entry[1]
        sub_sampling = 1
        suffix = '_bandw' + str(bandwidth) + '_RU' + str(sub_band) + '_sampling' + str(sub_sampling)

        name_file_save = './evaluations/' + args.names_base + '_' + str(csi_act) + '_' + suffix + '.txt'
        with open(name_file_save, "rb") as fp:  # Pickling
            metrics_matrix_dict = pickle.load(fp)

        accuracies_cross_val = metrics_matrix_dict['accuracies_cross_val']
        fscores_cross_val = metrics_matrix_dict['fscores_cross_val']

        avg_accuracies_cross_val[idx, :] = np.mean(accuracies_cross_val, axis=0)
        avg_accuracies_activities[idx, :] = np.mean(accuracies_cross_val, axis=1)

        avg_fscores_cross_val[idx, :] = np.mean(fscores_cross_val, axis=0)
        avg_fscores_activities[idx, :] = np.mean(fscores_cross_val, axis=1)

    stats_accuracies_cross_val = []
    stats_accuracies_activities = []
    stats_fscores_cross_val = []
    stats_fscores_activities = []
    for idx in range(n_entries):
        stats_accuracies_cross_val.append(cbook.boxplot_stats(avg_accuracies_cross_val[idx], whis=(5, 95))[0])
        stats_accuracies_activities.append(cbook.boxplot_stats(avg_accuracies_activities[idx], whis=(5, 95))[0])
        stats_fscores_cross_val.append(cbook.boxplot_stats(avg_fscores_cross_val[idx], whis=(5, 95))[0])
        stats_fscores_activities.append(cbook.boxplot_stats(avg_fscores_activities[idx], whis=(5, 95))[0])

    # plot accuracies
    stats = [stats_accuracies_cross_val, stats_accuracies_activities]
    stats_names = ['stats_accuracies_cross_val', 'stats_accuracies_activities']

    for idx_st, stat in enumerate(stats):
        fig, ax = plt.subplots(1, 1, constrained_layout=True)
        fig.set_size_inches(9, 2.5)
        # Plot boxplots from our computed statistics
        bp = ax.bxp(stat, positions=np.arange(n_entries), showfliers=False, widths=0.2)
        plt.setp(bp['boxes'], color='black', linewidth=1.5)
        plt.setp(bp['medians'], color='black', linewidth=1.5)
        plt.setp(bp['whiskers'], color='black')

        for box in bp['boxes']:
            box_x = []
            box_y = []
            for j in range(5):
                box_x.append(box.get_xdata()[j])
                box_y.append(box.get_ydata()[j])
            box_coords = np.column_stack([box_x, box_y])
            ax.add_patch(Polygon(box_coords, facecolor='C4', alpha=0.7))

        ax.set_xticklabels(band_subband_names)
        plt.grid(which='both')
        plt.ylim([0.3, 1])
        plt.yticks(np.linspace(0.3, 1, 8), np.linspace(30, 100, 8, dtype=int))
        plt.xlabel(r'resource unit')
        plt.ylabel(r'accuracy [$\%$]')
        name_fig = './plots/change_bw_' + stats_names[idx_st] + '.pdf'
        plt.savefig(name_fig)
        plt.close()

    # plot fscores
    stats = [stats_fscores_cross_val, stats_fscores_activities]
    stats_names = ['stats_fscores_cross_val', 'stats_fscores_activities']

    for idx_st, stat in enumerate(stats):
        fig, ax = plt.subplots(1, 1, constrained_layout=True)
        fig.set_size_inches(9, 2.5)
        # Plot boxplots from our computed statistics
        bp = ax.bxp(stat, positions=np.arange(n_entries), showfliers=False, widths=0.2)
        plt.setp(bp['boxes'], color='black', linewidth=1.5)
        plt.setp(bp['medians'], color='black', linewidth=1.5)
        plt.setp(bp['whiskers'], color='black')

        for box in bp['boxes']:
            box_x = []
            box_y = []
            for j in range(5):
                box_x.append(box.get_xdata()[j])
                box_y.append(box.get_ydata()[j])
            box_coords = np.column_stack([box_x, box_y])
            ax.add_patch(Polygon(box_coords, facecolor='C4', alpha=0.7))

        ax.set_xticklabels(band_subband_names)
        plt.grid(which='both')
        plt.ylim([0.3, 1])
        plt.yticks(np.linspace(0.3, 1, 8))
        plt.xlabel(r'resource unit')
        plt.ylabel(r'F1-score')
        name_fig = './plots/change_bw_' + stats_names[idx_st] + '.pdf'
        plt.savefig(name_fig)
        plt.close()

    # plot accuracy f-score together cross-val
    stats = [stats_accuracies_activities, stats_fscores_activities]

    fig, ax = plt.subplots(1, 1, constrained_layout=True)
    fig.set_size_inches(7.2, 2.5)
    # Plot boxplots from our computed statistics
    bp = ax.bxp(stats[0], positions=np.arange(n_entries) - 0.12, showfliers=False, widths=0.24,
                   manage_ticks=False)
    plt.setp(bp['boxes'], color='black', linewidth=1)
    plt.setp(bp['medians'], color='black', linewidth=1.5)
    plt.setp(bp['whiskers'], color='black')
    for box in bp['boxes']:
        box_x = []
        box_y = []
        for j in range(5):
            box_x.append(box.get_xdata()[j])
            box_y.append(box.get_ydata()[j])
        box_coords = np.column_stack([box_x, box_y])
        ax.add_patch(Polygon(box_coords, facecolor='C4', alpha=0.7))

    bp = ax.bxp(stats[1], positions=np.arange(n_entries) + 0.12, showfliers=False, widths=0.24,
                   manage_ticks=False)
    plt.setp(bp['boxes'], color='black', linewidth=1)
    plt.setp(bp['medians'], color='black', linewidth=1.5)
    plt.setp(bp['whiskers'], color='black')
    for box in bp['boxes']:
        box_x = []
        box_y = []
        for j in range(5):
            box_x.append(box.get_xdata()[j])
            box_y.append(box.get_ydata()[j])
        box_coords = np.column_stack([box_x, box_y])
        ax.add_patch(Polygon(box_coords, facecolor='C1'))

    ax.set_xticks(np.arange(n_entries))
    ax.set_xticklabels(band_subband_names)
    plt.grid(which='both')
    plt.ylim([0.3, 1])
    plt.yticks(np.linspace(0.3, 1, 8))
    plt.xlabel(r'resource unit')
    plt.ylabel(r'metric')
    custom_lines = [Line2D([0], [0], color='C4', linewidth=4, alpha=0.7),
                    Line2D([0], [0], color='C1', linewidth=4)]
    plt.legend(custom_lines, [r'accuracy', r'F1-score'],
               ncol=1, labelspacing=0.2, columnspacing=0.5, fontsize='medium')#, loc='lower right')
    name_fig = './plots/change_bw_accuracy_fscore_activities.pdf'
    plt.savefig(name_fig)
    plt.close()
