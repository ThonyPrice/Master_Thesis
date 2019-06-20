import pickle

import argparse
import matplotlib.pyplot as plt
import numpy as np
import plot_utils as pu
import pandas as pd


"""
Chapter: Methods

Description:
Visualize dynamics of insulin compartments for a specific subject.

Script:
python3 insulin_compartments.py --save ./eps/insulin_compartments.eps

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
    # Collect insulin data
    tissue_compartment = df['s1'].values[:N]
    plasma_compartment = df['s2'].values[:N]
    bolus_data = df['bolus_inputs_'].values[:N]

    # Keep only non-zero bolus data - Set zeros to nan
    bolus_data[bolus_data==0] = np.nan

    # Plot Insulin Compartment Data
    ax = fig.add_subplot(111)
    ax.plot(x, tissue_compartment, c='b', lw=pu.plot_lw(),
            label='Subcutaneous tissue')
    ax.plot(x, plasma_compartment, c='r', lw=pu.plot_lw(),
            label='Blood plasma')
    ax.set_xlabel('$t \, (minutes)$')
    ax.set_ylabel('$Insulin, \, mg/L$')
    ax.set_axisbelow(True)

    # Add Bolus Data on second y-axis
    ax2 = ax.twinx()
    color = 'tab:green'
    ax2.set_ylabel('Units', color=color)
    ax2.plot(x, bolus_data, #lw=pu.plot_lw(),
            label='Bolus', color=color, marker='D', markerfacecolor='None',
            markersize=5, linewidth=0)
    ax2.tick_params(axis='y', labelcolor=color)

    # Show legend
    ax.legend(loc='upper left', prop={'size': 5})
    ax2.legend(loc='upper right', prop={'size': 5})


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
