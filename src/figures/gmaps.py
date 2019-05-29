import pickle
import random

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from skimage import io


"""
Docstring...
"""

# File path to read CGM data from
F_NAME_DATA = './../data/minimal_tracking_data.2019-05-32.csv'

F_NAME_MODEL = './../results/model_henrik_2019-05-29-10:02.pkl'


def main():

    # Fetch data
    df = pd.read_csv(F_NAME_DATA)
    df_model = pd.read_pickle(path=F_NAME_MODEL)

    # Pick a user
    USR_IDS = df['user_id'].unique()
    # u_id = random.choice(USR_IDS)
    u_id = 'henrik'
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

    # idxs = idxs[:4]
    # urls = urls[:4]
    # glucose_chunks = glucose_chunks[:4]

    # Visualize a meal image
    n = len(urls)
    columns = 8
    rows = max((n//8)+1, 2)
    link = urls[0]
    f, axarr = plt.subplots(rows, columns)
    for i in range(n):
        # img = io.imread(urls[i])
        # axarr[i//columns,i%columns].imshow(img)
        axarr[i//columns,i%columns].plot(np.arange(glucose_len),
                                        glucose_chunks[i], c='orange')
    plt.show()


if __name__ == '__main__':
    main()
