import os
import pickle

import numpy as np
import pandas as pd


"""
Chapter: Results

Description:
Aggragate results of fixad-parameter-models into tables.

Script:
python3 aggregate_table.py
"""


RESULTS_DIR_1 = 'results/fixed_parameters_std-1'
RESULTS_DIR_2 = 'results/fixed_parameters_std-2'
RESULTS_DIR_3 = 'results/fixed_parameters_std-3'

TEX_SAVE_PATH = 'results/tables/results_summary.tex'

USERS = 14


def load_models_to_dict(directory, user_ids):
    """ Load user id's model data into individual dataframes.
        Arrange as a dictionary where the id is key and df the value.

    Keyword parameters:
    directory -- Path where the pickled model data is stored
    user_ids -- A list or which users' dataframes to include
    """

    d = {}
    for it, usr_id in enumerate(user_ids):
        f_name = '%s/%s.pkl'%(directory, usr_id)
        with open(f_name, 'rb') as f:
            d[it] = pickle.load(f)

    return d


def load_agg_scores_df(directory):

    file = 'agg_scores'
    f_name = '%s/%s.pkl'%(directory, file)
    with open(f_name, 'rb') as f:
        df = pickle.load(f)

    return df

def main():

    # user_list = np.arange(USERS)
    # models_data_dict = load_models_to_dict(RESULTS_DIR, user_list)

    result_dirs = [RESULTS_DIR_1, RESULTS_DIR_2, RESULTS_DIR_3]
    dfs = [load_agg_scores_df(dir) for dir in result_dirs]

    # Create individual data frames for each run
    for it, df in enumerate(dfs):
        # Clean up and organize
        # df['tmp_index'] = df.index
        # df.sort_values(['stds', 'tmp_index'], inplace=True)
        # df.drop(columns=['tmp_index'], inplace=True)

        df.loc['Mean'] = df.mean()
        df.loc['Std'] = df.std()
        df = df.round(2)

        df.to_latex(buf='tables/results_summary_%d.tex'%(it+1))

    # Create Aggregated scores Data Frame
    dfs = [load_agg_scores_df(dir) for dir in result_dirs]
    for it, df in enumerate(dfs):
        df.insert(loc=0, column='stds', value=it+1)

    df = dfs[0].append(dfs[1]).append(dfs[2])
    df = df.groupby(['stds'], as_index=False).agg(['mean','std'])
    df = df.round(2)
    df = df.T
    df.to_latex(buf='tables/results_summary.tex')


if __name__ == '__main__':
    main()
