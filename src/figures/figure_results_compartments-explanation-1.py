import pickle

import argparse
import matplotlib.pyplot as plt
import numpy as np
import plot_utils as pu
import pandas as pd


"""
Chapter: Results

Description:
Make two vertically aligned plots.
These display the given data, plots description.
Top -- Cgm and meals.
Bottom -- Insulin compartments and Boluses

Script:
python3 figure_results_compartments-explanation-0.py
        --save ./eps/figure-results-explain-1.eps

Source:
https://github.com/Wookai/paper-tips-and-tricks
"""


def main(args):

    # Collect data from results
    df = pd.read_pickle(path='results/fixed_parameters_std-2/4.pkl')

    # Set time range
    S = 540
    N = 710
    x = np.arange(N-S)

    # Prep figure
    pu.figure_setup()
    fig_size = pu.get_fig_size(10, 5)
    fig = plt.figure(figsize=(fig_size))


    ############################################################################
    # Collect CGM and meal data
    cgm_data = df['cgm_inputs_'].values[S:N]
    cgm_model = df['Gt'].values[S:N]
    meals = df['y_meals'].values[S:N]

    # Make all 0 target meal NaNs
    meals[meals==0] = np.nan

    # Print meal idxs
    print('Meal idxs: ', np.where(meals==1.))

    # Plot CGM and Meal data
    ax = fig.add_subplot(211)
    ax.set_xlim(0,N-S)
    ax.set_ylim(0,250)

    ax.axhline(70, linewidth=1, color='#e3e1e1')
    ax.axhline(180, linewidth=1, color='#e3e1e1')
    # ax.axvspan(125, 170, color='#eeeeee')

    ax.plot(x, cgm_data, marker='o', c='g', linewidth=0, markersize=1,
            label='CGM')

    ax.plot(x, meals*20, marker='o', c='purple', linewidth=0, markersize=4,
            markerfacecolor='None', label='Meal')
    ax.set_axisbelow(True)

    ax.legend(loc='upper left', prop={'size': 6})
    ax.set_ylabel('$mg/dl$')

    ############################################################################
    # Collect insulin data
    tissue_compartment = df['s1'].values[S:N]
    plasma_compartment = df['s2'].values[S:N]
    bolus_data = df['bolus_inputs_'].values[S:N]

    # Keep only non-zero bolus data - Set zeros to nan
    bolus_data[bolus_data==0] = np.nan

    # Print bolus idxs
    print('Bolus idxs: ', np.where(bolus_data>0.))

    # Plot Insulin Compartment Data
    ax = fig.add_subplot(212, sharex=ax)
    ax.set_xlim(0,N-S)

    ax.plot(x, tissue_compartment, c='b', lw=pu.plot_lw(),
            label='Subcutaneous tissue')
    ax.plot(x, plasma_compartment, c='r', lw=pu.plot_lw(),
            label='Blood plasma')

    ax.set_xlabel('$t \, (minutes)$')
    ax.set_ylabel('$Insulin, \, mg/L$')

    # plt.grid()
    ax.set_axisbelow(True)

    # Add Bolus Data on second y-axis
    ax2 = ax.twinx()
    color = 'tab:green'
    ax2.set_ylabel('Units', color=color)
    ax2.plot(x, bolus_data, #lw=pu.plot_lw(),
            label='Bolus', color=color, marker='D', markerfacecolor='None',
            markersize=4, linewidth=0)
    ax2.tick_params(axis='y', labelcolor=color)
    ax2.set_axisbelow(True)

    # Show legend
    ax.legend(loc='upper left', prop={'size': 6})
    ax2.legend(loc='upper right', prop={'size': 6})


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
