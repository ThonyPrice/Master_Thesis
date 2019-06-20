import pickle

import argparse
import matplotlib.pyplot as plt
import numpy as np
import plot_utils as pu
import pandas as pd


"""
Chapter: Results

Description:
Make two vertically aligned plots of all compartments to explain detection.
CGM, meals, predicted meals, standard deviation meal threshold.

Script:
python3 figure_results_compartments-explanation-3.py
        --save ./eps/figure-results-meal-detection.eps

Source:
https://github.com/Wookai/paper-tips-and-tricks
"""


def main(args):

    # Collect data from results
    df = pd.read_pickle(path='results/fixed_parameters_std-2/4.pkl')

    # Set time range
    S = 540
    N = 695
    H = 30
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
    print('Logged Meal idxs:', np.where(meals>0))

    # Get meal predictions
    meal_preds = df['meal_preds_'].values[S:N]
    meal_preds[meal_preds==0] = np.nan

    # Plot CGM and Meal data
    ax = fig.add_subplot(211)
    ax.set_xlim(0,N-S)
    ax.set_ylim(0,250)

    ax.axhline(70, linewidth=1, color='#ececec')
    ax.axhline(180, linewidth=1, color='#ececec')

    ax.plot(x, cgm_data, marker='o', c='g', linewidth=0, markersize=1,
            label='CGM')
    ax.plot(x, meals*20, marker='o', c='purple', linewidth=0, markersize=4,
            markerfacecolor='None', label='Meal')
    ax.plot(x, meal_preds*5, marker='^', c='magenta', linewidth=0,
            markersize=4, label='Predicted meal')

    ax.set_axisbelow(True)

    ax.legend(loc='upper left', prop={'size': 6})
    ax.set_ylabel('$mg/dl$')

    ############################################################################
    # Glucose and Meal Detection

    # Get glucose data
    carb_est = df['carb_ests_'].values[S:N]
    carb_mean = df['carb_mean_'].values[S:N]
    carb_stds = df['carb_stds_'].values[S:N]

    # Plot
    ax = fig.add_subplot(212, sharex=ax)
    ax.set_xlim(0,N-S)

    for it, m in enumerate(meal_preds):
        if m == 1:
            print('Meal prediction idx: ', it)
            ax.axvline(it, c='magenta')

    ax.plot(x, carb_est, c='orange', lw=pu.plot_lw(), label="$\hat{m}_t'$")
    ax.plot(x, carb_mean, c='b', lw=.8, label="$\\overline{\hat{m'}}_{0:t}$")
    ax.plot(x, carb_stds, c='b', lw=.8, label="$2 \cdot \sigma$", linestyle='-.')

    ax.legend(loc='upper left', prop={'size': 6})
    ax.set_xlabel('$t \, (minutes)$')
    ax.set_ylabel('$Carbs, \, mg$')
    ax.set_axisbelow(True)

    # plt.grid()
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
