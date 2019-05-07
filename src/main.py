import pickle
import time
import warnings

import numpy as np
import pandas as pd
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import KFold

from AggregatedMealModel import AggregatedMealModel
from DataStatistics import DataStatistics
from DisableCV import DisableCV
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
HP_PKL_PATH = './parameter_searches/search_%s.pkl'%\
                (time.strftime("%Y-%m-%d-%H:%M"))


def print_all_data_shapes(X, y):
    """Print all shapes of matrices and vectors contained in X and y."""

    for it, _ in enumerate(X_agg):
        print('User: %d | x-shape: (%d, %d) | y-shape: (%d, %d)'%\
            (it, X_agg[it].shape[0], X_agg[it].shape[1],
            y_agg[it].shape[0], y_agg[it].shape[1]))


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
    N = 2000

    # Aggregate data over users
    X_agg = [usr_df[['units', 'quantity']].values[:N,:] for usr_df in dfs]
    y_agg = [usr_df[['meal']].values[:N] for usr_df in dfs]
    # print_all_data_shapes(X_agg, y_agg)

    # GridSearch Params
    param_grid = {
            #'horizon': [30, 45],#[30, 45],
            'g': np.linspace(.001, .5, 5), #np.random.uniform(1./100, 1., 3),
            'h': np.linspace(.001, .5, 5), #np.random.uniform(1./100, 1., 3)
            'meal_threshold': np.linspace(2,10,2)
    }

    grid_search = GridSearchCV(
            estimator=AggregatedMealModel(),
            param_grid=param_grid,
            iid=False,
            cv=DisableCV(),
            refit=False,
            verbose=2)
    grid_search.fit(X_agg, y_agg)

    print('Best params:', grid_search.best_params_)
    print('Best score:', grid_search.best_score_)

    # Pickle CV results
    with open(HP_PKL_PATH, 'wb') as handle:
        pickle.dump(grid_search, handle, protocol=pickle.HIGHEST_PROTOCOL)

    # Run with best params
    bestModel = AggregatedMealModel(**grid_search.best_params_)
    bestModel.fit(X_agg, y_agg)
    bestModel.score(X_agg, y_agg, mode='mean', verbose=1)

    print('*** EOF ***')


if __name__ == '__main__':
    main()
