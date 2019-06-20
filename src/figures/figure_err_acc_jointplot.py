import argparse
import pickle

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import plot_utils as pu
import seaborn as sns


"""
Chapter: Results

Description:
Investigate correlation between model error and model scores by jointplots.

Script:
python3 figure_err_acc_jointplot.py --save ./eps/err-scores-correlation.eps

Source:
https://github.com/Wookai/paper-tips-and-tricks
"""


RESULTS_DIR_1 = 'results/fixed_parameters_std-1'

USERS = 14

def load_agg_scores_df(directory):

    file = 'agg_scores'
    f_name = '%s/%s.pkl'%(directory, file)
    with open(f_name, 'rb') as f:
        df = pickle.load(f)

    return df

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
            d[usr_id] = pickle.load(f)

    return d


def calc_err(df):
    """ Calculate model prediction error by comparing measured and modeled CGM
    """

    y = df['cgm_values_'].values
    y_hat = df['Gt'].values

    assert(len(y) == len(y_hat))
    calibration_time = 3*60

    res = (y-y_hat)
    res = res[calibration_time:-calibration_time]
    mse = res.mean()

    return abs(y-y_hat).mean()


def main(args):

    usr_ids = np.arange(USERS)
    usr_scores = [0]*USERS
    for it, usr_id in enumerate(usr_ids):

        # Collect metrics (TPR, PPV, FDR)
        agg_df = load_agg_scores_df(RESULTS_DIR_1)
        score_metrics = agg_df[agg_df.index == usr_id]
        score_metrics = score_metrics[['TPR', 'PPV', 'FDR']]

        # Calc ERR
        usr_dict = load_models_to_dict(directory=RESULTS_DIR_1, user_ids=[usr_id])
        usr_df = usr_dict[usr_id]
        score_metrics['ERR'] = calc_err(usr_df)
        usr_scores[it] = score_metrics

    df = pd.concat(usr_scores, axis=0)
    print(df)


    pu.figure_setup()

    fig_size = pu.get_fig_size(10, 4)
    fig = plt.figure(figsize=(fig_size))

    ax = fig.add_subplot(131)
    ax.plot(df['TPR'], df['ERR'], c='b', lw=0, marker='o', markerfacecolor=None,
            markersize=3)
    ax.set_ylabel('$ERR$')
    ax.set_xlabel('$TPR$')
    plt.grid()
    ax.set_axisbelow(True)

    ax = fig.add_subplot(132, sharex=ax, sharey=ax)
    ax.plot(df['PPV'], df['ERR'], c='b', lw=0, marker='o', markerfacecolor=None,
            markersize=3)
    ax.set_xlabel('$PPV$')
    plt.grid()
    ax.set_axisbelow(True)

    ax = fig.add_subplot(133, sharex=ax, sharey=ax)
    ax.plot(df['FDR'], df['ERR'], c='b', lw=0, marker='o', markerfacecolor=None,
            markersize=3)
    ax.set_xlabel('$FDR$')
    plt.grid()
    ax.set_axisbelow(True)

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
