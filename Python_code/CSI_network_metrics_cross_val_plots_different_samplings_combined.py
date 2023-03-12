
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
    parser.add_argument('names_base1', help='Names base for the files')
    parser.add_argument('names_base2', help='Names base for the files diff subsamples')
    args = parser.parse_args()

    names_string1 = args.names_base1
    names_string2 = args.names_base2
    names_files1 = []
    names_files2 = []
    for nam in names_string1.split(','):
        names_files1.append(nam)
    for nam in names_string2.split(','):
        names_files2.append(nam)
    names_string1 = ','.join(names_files1)
    names_string2 = ','.join(names_files2)
    names_string = [names_string1, names_string2]
    names_files1 = np.asarray(names_files1)
    names_files2 = np.asarray(names_files2)
    num_files = names_files1.shape[0]

    csi_act = args.activities
    activities = []
    for lab_act in csi_act.split(','):
        activities.append(lab_act)
    activities = np.asarray(activities)
    num_act = activities.shape[0]

    #################################
    # BOX PLOT DIFFERENT SAMPLINGS
    #################################
    samplings = np.arange(1, 6)
    n_entries = samplings.shape[0]

    avg_accuracies_cross_val = np.zeros((num_files, num_act))
    avg_fscores_cross_val = np.zeros((num_files, num_act))

    num_cross_val = 12
    avg_accuracies_activities = np.zeros((num_files, num_cross_val*num_files))
    avg_fscores_activities = np.zeros((num_files, num_cross_val*num_files))

    positions1 = [0, 1, 3, 5, 7]
    indices1 = np.arange(len(positions1))
    positions2 = [2, 4, 6, 8]
    indices2 = np.arange(len(positions2)) + 1
    positions_list = [positions1, positions2]
    indices_list = [indices1, indices2]
    for idx_s, name in enumerate(names_string):
        positions = positions_list[idx_s]
        indices = indices_list[idx_s]
        for idx_ in range(len(positions)):
            idx_plot = positions[idx_]
            idx = indices[idx_]
            print(idx_plot)
            bandwidth = 80
            sub_band = 1
            sub_sampling = idx + 1
            suffix = '_bandw' + str(bandwidth) + '_RU' + str(sub_band) + '_sampling' + str(sub_sampling)

            name_file_save = './evaluations/' + name + '_' + str(csi_act) + '_' + suffix + '.txt'
            try:
                with open(name_file_save, "rb") as fp:  # Pickling
                    metrics_matrix_dict = pickle.load(fp)
            except FileNotFoundError:
                print(name_file_save, ' not found')
                continue

            accuracies_cross_val = metrics_matrix_dict['accuracies_cross_val']
            fscores_cross_val = metrics_matrix_dict['fscores_cross_val']

            avg_accuracies_cross_val[idx_plot, :] = np.mean(accuracies_cross_val, axis=0)
            avg_accuracies_activities[idx_plot, :] = np.mean(accuracies_cross_val, axis=1)

            avg_fscores_cross_val[idx_plot, :] = np.mean(fscores_cross_val, axis=0)
            avg_fscores_activities[idx_plot, :] = np.mean(fscores_cross_val, axis=1)

    stats_accuracies_cross_val = []
    stats_accuracies_activities = []
    stats_fscores_cross_val = []
    stats_fscores_activities = []
    for idx in range(num_files):
        stats_accuracies_cross_val.append(cbook.boxplot_stats(avg_accuracies_cross_val[idx], whis=(5, 95))[0])
        stats_accuracies_activities.append(cbook.boxplot_stats(avg_accuracies_activities[idx], whis=(5, 95))[0])
        stats_fscores_cross_val.append(cbook.boxplot_stats(avg_fscores_cross_val[idx], whis=(5, 95))[0])
        stats_fscores_activities.append(cbook.boxplot_stats(avg_fscores_activities[idx], whis=(5, 95))[0])

    # plot accuracies
    stats = [stats_accuracies_cross_val, stats_accuracies_activities]
    stats_names = ['stats_accuracies_cross_val', 'stats_accuracies_activities']

    labels = [r'$\left(T_c,N\right)$',
              r'$\left(\frac{T_c}{2},N\right)$', r'$\left(\frac{T_c}{2},\frac{N}{2}\right)$',
              r'$\left(\frac{T_c}{3},N\right)$', r'$\left(\frac{T_c}{3},\frac{N}{3}\right)$',
              r'$\left(\frac{T_c}{4},N\right)$', r'$\left(\frac{T_c}{4},\frac{N}{4}\right)$',
              r'$\left(\frac{T_c}{5},N\right)$', r'$\left(\frac{T_c}{5},\frac{N}{5}\right)$']

    for idx_st, stat in enumerate(stats):
        fig, ax = plt.subplots(1, 1, constrained_layout=True)
        fig.set_size_inches(9, 2.5)
        # Plot boxplots from our computed statistics
        bp = ax.bxp(stat, positions=np.arange(num_files), showfliers=False, widths=0.2)
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

        ax.set_xticklabels(labels, rotation=0)
        plt.grid(which='both')
        plt.ylim([0.2, 1])
        plt.yticks(np.linspace(0.2, 1, 9), np.linspace(20, 100, 9, dtype=int))
        plt.xlabel(r'sampling')
        plt.ylabel(r'accuracy [$\%$]')
        name_fig = './plots/change_sampl_' + stats_names[idx_st] + '_combined.pdf'
        plt.savefig(name_fig)
        plt.close()

    # plot fscores
    stats = [stats_fscores_cross_val, stats_fscores_activities]
    stats_names = ['stats_fscores_cross_val', 'stats_fscores_activities']

    for idx_st, stat in enumerate(stats):
        fig, ax = plt.subplots(1, 1, constrained_layout=True)
        fig.set_size_inches(9, 2.5)
        # Plot boxplots from our computed statistics
        bp = ax.bxp(stat, positions=np.arange(num_files), showfliers=False, widths=0.2)
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

        ax.set_xticklabels(labels, rotation=0)
        plt.grid(which='both')
        plt.ylim([0.2, 1])
        plt.yticks(np.linspace(0.2, 1, 9))
        plt.xlabel(r'sampling')
        plt.ylabel(r'F1-score')
        name_fig = './plots/change_sampl_' + stats_names[idx_st] + '_combined.pdf'
        plt.savefig(name_fig)
        plt.close()

    # plot accuracy f-score together cross-val
    stats = [stats_accuracies_activities, stats_fscores_activities]

    fig, ax = plt.subplots(1, 1, constrained_layout=True)
    fig.set_size_inches(7.2, 2.9)
    # Plot boxplots from our computed statistics
    bp = ax.bxp(stats[0], positions=np.arange(num_files) - 0.15, showfliers=False, widths=0.30,
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

    bp = ax.bxp(stats[1], positions=np.arange(num_files) + 0.15, showfliers=False, widths=0.30,
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

    ax.set_xticks(np.arange(num_files))
    ax.set_xticklabels(labels, rotation=0)
    plt.grid(which='both')
    plt.ylim([0.2, 1])
    plt.yticks(np.linspace(0.2, 1, 9))
    plt.xlabel(r'sampling')
    plt.ylabel(r'metric')
    custom_lines = [Line2D([0], [0], color='C4', linewidth=4, alpha=0.7),
                    Line2D([0], [0], color='C1', linewidth=4)]
    plt.legend(custom_lines, [r'accuracy', r'F1-score'],
               ncol=2, labelspacing=0.2, columnspacing=0.5, fontsize='medium', loc='lower center')
    name_fig = './plots/change_sampl_accuracy_fscore_activities_combined.pdf'
    plt.savefig(name_fig)
    plt.close()
