import pickle

import argparse
import matplotlib.pyplot as plt
import numpy as np
import plot_utils as pu
import pandas as pd


"""
Chapter: Methods

Description:
Visualize dynamics of glucose compartments for a specific subject.

Script:
python3 glucose_compartments.py --save ./eps/glucose_compartments.eps

Source:
https://github.com/Wookai/paper-tips-and-tricks
"""


def main(args):

    # Collect data from results
    df = pd.read_pickle(path='results/fixed_parameters_std-2/8.pkl')

    # Set time range, 1440 = 1 day
    N = 1440
    x = np.arange(N)

    # Prep figure
    pu.figure_setup()
    fig_size = pu.get_fig_size(10, 5)
    fig = plt.figure(figsize=(fig_size))

    ############################################################################
    # Collect glucose data
    gut_compartment = df['gt'].values[:N]
    plasma_compartment = df['mt'].values[:N]
    carb_est = df['carb_ests_'].values[:N]
    # carb_est[carb_est==0] = np.nan

    # Plot Glucose compartments...
    ax = fig.add_subplot(111)
    ax.plot(x, gut_compartment, c='b', lw=pu.plot_lw(), label='Gut glucose')
    ax.plot(x, plasma_compartment, c='r', lw=pu.plot_lw(), label='Plasma glucose')
    ax.set_xlabel('$t \, (minutes)$')
    ax.set_ylabel('$mg/L$')
    ax.set_axisbelow(True)

    # Add second axis
    ax2 = ax.twinx()
    color = 'tab:orange'
    ax2.set_ylabel('$\hat{m}$', color=color)
    ax2.plot(x, carb_est, #lw=pu.plot_lw(),
            label='Carbs est.', color=color, alpha=.5)#, marker='o', markerfacecolor='None',
            #markersize=5, linewidth=0, alpha=.2)
    ax2.tick_params(axis='y', labelcolor=color)


    # Add legends
    ax.legend(loc='upper left', prop={'size': 5})
    ax2.legend(loc='upper right', prop={'size': 5})

    # Finilize plot
    plt.grid()
    plt.tight_layout()

    if args.save:
        pu.save_fig(fig, args.save)
    else:
        plt.show()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('-s', '--save')

    args = parser.parse_args()
    main(args)
