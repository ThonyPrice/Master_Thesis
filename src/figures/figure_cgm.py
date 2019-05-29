import argparse
import random

import matplotlib.pyplot as plt
import numpy as np
import plot_utils as pu
import pandas as pd
from scipy.interpolate import pchip_interpolate
from scipy.signal import savgol_filter

# Script:
# python3 figure_cgm.py --save ./eps/plain_cgm.eps
# python3 figure_cgm.py --save ./png/plain_cgm.png


def main(args):
    """Visualize basic CGM - This can be used as a template to add
    more coupled data."""

    # Load data
    F_NAME = './../data/minimal_tracking_data.2019-05-32.csv'
    df = pd.read_csv(F_NAME)

    # Pick a user
    random.seed(42)
    usr_id = random.choice(df['user_id'].unique())
    df = df[df['user_id'] == usr_id]

    # Set dateTimetimeIndex
    df['date'] = df['date'].apply(lambda x: x[:19])
    df['date'] = pd.to_datetime(df['date'])
    df.set_index('date', inplace=True)

    # Pick a day
    date = random.choice(df.index.date)
    df = df[date.strftime('%Y-%m-%d')]

    # Original CGM
    original_x_axis = np.arange(0,60*24,5)
    original_cgm_data = df['quantity']

    # Increase resolution by interpolation
    new_x_axis = np.arange(0,60*24,1)
    new_cgm_vals = pchip_interpolate(original_x_axis,
            original_cgm_data, new_x_axis)

    smoothed_cgm = savgol_filter(new_cgm_vals,
                             window_length=15,
                             polyorder=1,
                             mode='interp')

    # Remove overlap of indices
    # new_x_axis = new_x_axis[~original_x_axis]
    # new_cgm_vals = new_cgm_vals[~original_x_axis]

    # Prepare figure
    pu.figure_setup()
    fig_size = pu.get_fig_size(10, 5)
    fig = plt.figure(figsize=(fig_size))

    # Plot
    ax = fig.add_subplot(111)
    ax.axhspan(70, 180, alpha=0.1, color='black')
    ax.set_ylim(0,350)
    ax.plot(original_x_axis, original_cgm_data, marker='o', c='g',
            linewidth=0, markersize=1
            )
    ax.plot(new_x_axis, new_cgm_vals, marker='o', c='g', alpha=.2,
            linewidth=0, markersize=1
            )
    # ax.plot(new_x_axis, smoothed_cgm, c='black', alpha=.3,
    #         linestyle='dashed'
    #         )
    ax.set_xlabel('$t (minutes)$')
    ax.set_ylabel('$ mg/dl$')

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
