import pickle

import argparse
import matplotlib.pyplot as plt
import numpy as np
import plot_utils as pu
import pandas as pd


"""
Chapter: Results

Description:
Make four vertically aligned plots of all compartments to explain the dynamics.
CGM, Insulin, Glucose och Carbohydrate Estimation.

Script:
python3 figure_results_compartments-explanation-2.py
        --save ./eps/figure_all-compartments-prediction.eps

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
    fig_size = pu.get_fig_size(10, 8)
    fig = plt.figure(figsize=(fig_size))


    ############################################################################

    # Collect CGM and meal data
    cgm_data = df['cgm_inputs_'].values[S:N]
    cgm_model = df['Gt'].values[S:N]
    meals = df['y_meals'].values[S:N]

    # Set up prediction line
    pred_t = 124
    pred = df['cgm_preds_'][pred_t]
    x_pred = np.arange(pred_t, pred_t+len(pred))

    # Make all 0 target meal NaNs
    meals[meals==0] = np.nan

    # Print meal idxs
    print('Meal idxs: ', np.where(meals==1.))

    # Plot CGM and Meal data
    ax = fig.add_subplot(411)
    ax.set_xlim(0,N-S)
    ax.set_ylim(0,250)

    ax.axhline(70, linewidth=1, color='#ececec')
    ax.axhline(180, linewidth=1, color='#ececec')
    ax.axvspan(pred_t, pred_t+H+1, color='#f6f6f6')

    ax.plot(x, cgm_data, marker='o', c='g', linewidth=0, markersize=1,
            label='CGM')
    ax.plot(x_pred, pred, c='black', linewidth=1, linestyle='dashed',
            label='Modeled CGM')
    ax.plot(x, meals*20, marker='o', c='purple', linewidth=0, markersize=4,
            markerfacecolor='None', label='Meal')
    plt.vlines(x=pred_t+H, ymin=pred[-1], ymax=cgm_data[pred_t+H-1],
            color='r', linewidth=2)

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
    ax = fig.add_subplot(412, sharex=ax)
    ax.set_xlim(0,N-S)
    ax.axvspan(pred_t, pred_t+H+1, color='#f6f6f6')

    ax.plot(x, tissue_compartment, c='b', lw=pu.plot_lw(),
            label='Subcutaneous tissue')
    ax.plot(x, plasma_compartment, c='r', lw=pu.plot_lw(),
            label='Blood plasma')

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


    ############################################################################

    # Collect glucose data
    gut_compartment = df['gt'].values[S:N-H]
    plasma_compartment = df['mt'].values[S:N-H]
    xx = np.arange(N-S-H)

    # Collect "prediction" data
    gut_compartment_ff = df['gt'].values[N-H:N]
    plasma_compartment_ff = df['mt'].values[N-H:N]
    xx_ff = np.arange(N-S-H,N-S)

    # Plot Glucose compartments...
    ax = fig.add_subplot(413, sharex=ax)
    ax.set_xlim(0,N-S)
    ax.axvspan(pred_t, pred_t+H+1, color='#f6f6f6')

    ax.plot(xx, gut_compartment, c='b', lw=pu.plot_lw(), label='Gut glucose')
    ax.plot(xx, plasma_compartment, c='r', lw=pu.plot_lw(), label='Plasma glucose')

    ax.plot(xx_ff, gut_compartment_ff, c='b', lw=pu.plot_lw(),
            linestyle='dashed')
    ax.plot(xx_ff, plasma_compartment_ff, c='r', lw=pu.plot_lw(),
            linestyle='dashed')

    ax.set_ylabel('$mg/L$')
    ax.set_axisbelow(True)
    ax.legend(loc='upper left', prop={'size': 6})


    ############################################################################
    # Glucose and Meal Detection

    # Get glucose data
    carb_est = df['carb_ests_'].values[S:N-H]
    carb_mean = df['carb_mean_'].values[S:N-H]
    carb_stds = df['carb_stds_'].values[S:N-H]
    xx_carbs = np.arange(N-S-H)

    # Get "future" glucose data
    carb_est_ff = df['carb_ests_'].values[N-H:N]
    xx_carbs_ff = np.arange(N-S-H,N-S)

    # Get meal predictions
    meal_preds = df['meal_preds_'].values[S:N-H]
    meal_preds[meal_preds==0] = np.nan

    # Plot
    ax = fig.add_subplot(414, sharex=ax)
    ax.set_xlim(0,N-S)
    ax.axvspan(pred_t, pred_t+H+1, color='#f6f6f6')

    ax.plot(xx_carbs, carb_est, c='orange', lw=pu.plot_lw(), label='Carbs est.')
    ax.plot(xx_carbs_ff, carb_est_ff, c='orange', lw=pu.plot_lw(),
            linestyle='dashed')

    # ax.plot(x, carb_mean, c='b', lw=0.5, label)
    # ax.plot(xx_carbs, meal_preds, marker='^', c='magenta', linewidth=0,
    #         markersize=3, label='Predicted meal')

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
