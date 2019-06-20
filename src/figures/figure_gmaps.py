import argparse
import pickle
import random

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from skimage import io

import plot_utils as pu


"""
Chapter: Discussion

Description:
Generate food images and carb estimation images.
These can be overlayed and used ti argue that the system can provide gMaps.
Glucose Meal Action Profiles.

Script:
python3 figure_gmaps.py -s1 ./png/gmaps_im.png -s2 ./png/gmaps_plt.png

Source:
https://github.com/Wookai/paper-tips-and-tricks
"""

# File path to read CGM data from
F_NAME_DATA = 'data/minimal_tracking_data.csv'

F_NAME_MODEL = 'results/fixed_parameters_std-1/8.pkl'


def main(args):

    # Fetch data
    df = pd.read_csv(F_NAME_DATA)
    df_model = pd.read_pickle(path=F_NAME_MODEL)

    # Pick a user
    USR_IDS = df['user_id'].unique()
    # u_id = random.choice(USR_IDS)
    u_id = 8
    df = df[df['user_id'] == u_id]

    # Get idxs of all logged meals
    df.reset_index(inplace=True)
    meal_idxs = df.index[df['meal'] == 1].tolist()
    df = df.iloc[meal_idxs]
    df = df[df['image_url'] != '0']

    glucose_len = 38
    idxs = df.index.values
    urls = df['image_url'].values
    glucose_chunks = [df_model['carb_ests_'].values[idx:idx+glucose_len]
                     for idx in idxs]

    # Visualize a meal image
    n = len(urls)
    columns = 8
    rows = max((n//8)+1, 2)
    link = urls[0]
    fig, axarr = plt.subplots(rows, columns, sharex=True, sharey=True)
    for i in range(n):
        img = io.imread(urls[i])
        axarr[i//columns,i%columns].imshow(img)

    pu.save_fig(fig, args.im_save)
    plt.clf()

    fig, axarr = plt.subplots(rows, columns, sharex=True, sharey=True)
    for i in range(n):
        axarr[i//columns,i%columns].plot(np.arange(glucose_len),
                                        glucose_chunks[i], c='orange')
    pu.save_fig(fig, args.plt_save)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('-s1', '--im_save')
    parser.add_argument('-s2', '--plt_save')

    args = parser.parse_args()
    main(args)
