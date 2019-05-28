import argparse
import random

import matplotlib.pyplot as plt
import numpy as np
import plot_utils as pu
import pandas as pd
from scipy.interpolate import pchip_interpolate


# Script:
# python3 figure_various_meal_impacts.py --save ./meal_responses.eps


def main(args):
    """Plot multiple meals to illustrate the variance in meal responses."""

    # Load data
    F_NAME = 'data/minimal_tracking_data.2019-05-32.csv'
    df = pd.read_csv(F_NAME)

    # Set window length of meal response
    frequency = 5
    w_len = (2*60)//frequency
    x = np.arange(w_len)

    # Pick a user
    random.seed(42)
    usr_id = random.choice(df['user_id'].unique())
    df = df[df['user_id'] == usr_id]
    df.reset_index(inplace=True)

    # Prep data - Select frames only just after a meal is logged
    meal_chunks = []
    meal_idxs = df.index[df['meal'] == 1].tolist()

    for meal_idx in meal_idxs:
        meal_chunks.append(df.loc[meal_idx:meal_idx+w_len-1]['quantity'].values)

    # Normalize to have every line start from 0
    for it, chunk in enumerate(meal_chunks):
        delta = chunk[0]
        meal_chunks[it] = [x-delta for x in chunk]


    # Increase resolution by interpolation
    x_axis = np.arange(0,w_len*frequency,frequency)
    new_x_axis = np.arange(w_len*frequency)+1

    meal_chunks = [pchip_interpolate(
        x_axis, chunk, new_x_axis)
        for chunk in meal_chunks]

    # Prepare figure
    pu.figure_setup()
    fig_size = pu.get_fig_size(10, 5)
    fig = plt.figure(figsize=(fig_size))

    # Plot
    ax = fig.add_subplot(111)
    for meal in meal_chunks:
        ax.plot(new_x_axis, meal, c='b', lw=pu.plot_lw(), alpha=.3)
    ax.set_xlabel('$t (minutes)$')
    ax.set_ylabel('$\Delta mg/dl$')

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
