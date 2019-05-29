import pickle

import argparse
import matplotlib.pyplot as plt
import numpy as np
import plot_utils as pu
import pandas as pd


def main(args):

    # Collect data from results
    df = pd.read_pickle(path='./../results/m-test.pkl')

    with open('./../results/m-test-1-pred.pkl', 'rb') as f:
        cgm_preds = pickle.load(f)

    # Set time range, 1440 = 1 day
    N = 1440
    x = np.arange(N)

    # Prep figure
    pu.figure_setup()
    fig_size = pu.get_fig_size(10, 5)
    fig = plt.figure(figsize=(fig_size))


    ############################################################################
    # Collect CGM and meal data
    cgm_data = df['cgm_inputs_'].values[:N]
    cgm_model = df['Gt'].values[:N]
    meals = df['y_meals'].values[:N]

    # Make all 0 target meal NaNs
    meals[meals==0] = np.nan

    # Plot CGM and Meal data
    ax = fig.add_subplot(411)
    ax.axhspan(70, 180, alpha=0.1, color='black')
    ax.set_ylim(0,350)
    ax.plot(x, cgm_data, marker='o', c='g', linewidth=0, markersize=1, label='CGM')
    ax.plot(x, cgm_model, c='black', linewidth=1, linestyle='dashed', label='Modeled CGM')
    ax.plot(x, meals*20, marker='*', c='purple', linewidth=0, markersize=4, label='Meal')
    #ax.set_xlabel('$t (minutes)$')
    ax.legend(loc='upper left', prop={'size': 6})
    ax.set_ylabel('$mg/dl$')

    meal_preds = df['meal_preds_'].values[:N]
    meal_preds[meal_preds==0] = np.nan

    for it, m in enumerate(meal_preds):
        if m == 1:
            ax.axvline(it, c='magenta', alpha=.1)

    # Add CGM prediction
    # horizon = 45
    # start_time = 488
    # x_pred = np.arange(start_time,start_time+horizon+1)
    # cgm_pred = cgm_preds[start_time]-46
    # ax.plot(x_pred, cgm_pred, c='black', linewidth=1, linestyle='dashed')

    ############################################################################
    # Collect insulin data
    tissue_compartment = df['s1'].values[:N]
    plasma_compartment = df['s2'].values[:N]
    bolus_data = df['bolus_inputs_'].values[:N]

    # Keep only non-zero bolus data - Set zeros to nan
    bolus_data[bolus_data==0] = np.nan

    # Plot Insulin Compartment Data
    ax = fig.add_subplot(412)
    ax.plot(x, tissue_compartment, c='b', lw=pu.plot_lw(),
            label='Subcutaneous tissue')
    ax.plot(x, plasma_compartment, c='r', lw=pu.plot_lw(),
            label='Blood plasma')
    # ax.set_xlabel('$t \, (minutes)$')
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
    ax.legend(loc='upper left', prop={'size': 6})
    # ax2.legend(loc='upper right', prop={'size': 6})


    ############################################################################
    # Collect glucose data
    gut_compartment = df['gt'].values[:N]
    plasma_compartment = df['mt'].values[:N]

    # Plot Glucose compartments...
    ax = fig.add_subplot(413)
    ax.plot(x, gut_compartment, c='b', lw=pu.plot_lw(), label='Gut glucose')
    ax.plot(x, plasma_compartment, c='r', lw=pu.plot_lw(), label='Plasma glucose')
    # ax.set_xlabel('$t \, (minutes)$')
    ax.set_ylabel('$mg/L$')
    ax.set_axisbelow(True)
    ax.legend(loc='upper left', prop={'size': 6})

    ############################################################################
    # Glucose and Meal Detection

    # Get glucose data
    carb_est = df['carb_ests_'].values[:N]
    carb_mean = df['carb_mean_'].values[:N]
    carb_stds = df['carb_stds_'].values[:N]

    # Get meal predictions
    meal_preds = df['meal_preds_'].values[:N]
    meal_preds[meal_preds==0] = np.nan

    # Plot
    ax = fig.add_subplot(414)
    ax.plot(x, carb_est, c='orange', lw=pu.plot_lw())
    # ax.plot(x, carb_mean, c='b', lw=pu.plot_lw(), linestyle='dashed')
    # ax.plot(x, carb_mean, c='b', lw=pu.plot_lw(), linestyle='dashed')
    ax.plot(x, meal_preds*carb_est, marker='*', c='magenta', linewidth=0, markersize=4,
            label='Predicted meal')
    ax.set_xlabel('$t \, (minutes)$')
    ax.set_ylabel('$Carbs, \, mg$')
    ax.set_axisbelow(True)

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
