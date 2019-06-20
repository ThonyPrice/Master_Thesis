import argparse
import json
import os
import pickle
import time
import warnings

import numpy as np
import pandas as pd
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import KFold

from AggregatedMealModel import AggregatedMealModel
from MealPredictionModel import MealPredictionModel
from MinimalBergmanModel import BergmanModel


# Turn off warning caused by old MacOS driver
warnings.filterwarnings(action="ignore",
                        module="scipy",
                        message="^internal gelsd")

# Column names for Data in imported CSV
CGM_COL_NAME = 'quantity'
INSULIN_COL_NAME = 'units'
MEAL_COL_NAME = 'meal'

# File path to read CGM data from
F_NAME = 'data/minimal_tracking_data.csv'

# Add experiment description to results folder
def experiment_description(path, params):
    with open(path + '/_params.txt', 'w') as f:
        print('Experiment Parameters:\n', file=f)
        for p in params: print('%s: %d'%(p, params[p]), file=f)


def main(args):

    # Read arguments and set up experiment directories
    PARAMS = json.loads(args.params)
    RESULTS_PATH = args.save
    try:
        os.mkdir(RESULTS_PATH)
    except:
        print('Directory %s already exists')
    experiment_description(RESULTS_PATH, PARAMS)

    # Fetch data
    df = pd.read_csv(F_NAME)

    # Structure Data from DataFrame
    USR_IDS = df['user_id'].unique()
    dfs = df.groupby('user_id')
    dfs = [dfs.get_group(u_id) for u_id in USR_IDS]
    data = [np.hstack([
            df[INSULIN_COL_NAME].values.reshape(-1,1).astype(float),
            df[CGM_COL_NAME].values.reshape(-1,1).astype(float),
            df[MEAL_COL_NAME].values.reshape(-1,1)]).astype(float)
            for df in dfs]

    # Prep dictionary to keep all scores
    agg_scores = {}

    # Evaluate each user
    for it, user_data in enumerate(data):

        # Split data
        X, y = user_data[:,:2], user_data[:,2]

        # Grab user id and mk storage path
        u_id = USR_IDS[it]
        store_model_path = '%s/%s.pkl'%(RESULTS_PATH, u_id)

        # Run Model
        M = MealPredictionModel(**PARAMS)
        M.fit(X)
        M.score(y, verbose=True)
        M.store_model(path=store_model_path)

        agg_scores[u_id] = M.scores_

    # Save aggregated scores
    f_name = '%s/agg_scores.pkl'%(RESULTS_PATH)
    agg_scores_df = pd.DataFrame.from_dict(agg_scores, orient='index')
    with open(f_name, 'wb') as f:
        pickle.dump(agg_scores_df, f)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('-p', '--params')
    parser.add_argument('-s', '--save')

    args = parser.parse_args()
    main(args)
