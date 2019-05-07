import pickle
import time
import warnings

import numpy as np
import pandas as pd
from sklearn.model_selection import GridSearchCV

from AggregatedMealModel import AggregatedMealModel
from DataStatistics import DataStatistics
from MealPredictionModel import MealPredictionModel
from MinimalBergmanModel import BergmanModel
from PreProcess import PreProcess


"""
# TODO:

-- Hyper parameter search
-- Compartments plot
-- Score/evaluation plots

"""

# Turn off warning caused by old MacOS driver
warnings.filterwarnings(    action="ignore",
                            module="scipy",
                            message="^internal gelsd")

# File path to read CGM data from
F_NAME = 'data/minimal_tracking_data.2019-04-08.csv'

# File path to store Latex tables to
TEX_PATH = '/latex_outputs'

# File path to store hyper parameter serach results
HP_PKL_PATH = './parameter_searches/search_%s.pkl'%(time.strftime("%Y-%m-%d-%H:%M"))

def main():

    # Fetch data
    df = pd.read_csv(F_NAME)

    # Set dateTimetimeIndex
    df['date'] = df['date'].apply(lambda x: x[:19])
    df['date'] = pd.to_datetime(df['date'])
    df.set_index('date', inplace=True)

    # Print data stats
    #DataStats = DataStatistics(df)
    #print(DataStats.summary())
    #DataStats.to_tex(TEX_PATH)

    # Drop user_id 6 - Not enough data
    df = df[df['user_id'] != 6]

    # Preprocess dataFrames
    PreProcessPipe = PreProcess()

    USR_IDS = df['user_id'].unique()
    dfs = df.groupby('user_id')
    dfs = [dfs.get_group(u_id) for u_id in USR_IDS]
    dfs = [df.sort_index() for df in dfs]
    dfs = [PreProcessPipe.apply(df) for df in dfs]

    # Cap data size while testing
    N = 400

    # Aggregate data over users
    X_agg = [usr_df[['units', 'quantity']].values[:N,:] for usr_df in dfs]
    y_agg = [usr_df[['meal']].values[:N] for usr_df in dfs]

    # GridSearch Params
    param_grid = {
        'horizon': [30],#[30, 45],
        'g': [.5],#np.random.uniform(1./100, 1., 3),
        'h': [.01]#np.random.uniform(1./100, 1., 3)
    }

    grid_search = GridSearchCV(
            estimator=AggregatedMealModel(),
            param_grid=param_grid,
            cv=2, # There's no randomness to this Model.
            verbose=2)
    grid_search.fit(X_agg, y_agg)

    print('Best params:', grid_search.best_params_)
    print('Best score:', grid_search.best_score_)

    # Pickle CV results
    with open(HP_PKL_PATH, 'wb') as handle:
        pickle.dump(grid_search, handle, protocol=pickle.HIGHEST_PROTOCOL)

    print('*** EOF ***')

if __name__ == '__main__':
    main()
