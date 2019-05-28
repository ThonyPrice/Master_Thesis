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


# Turn off warning caused by old MacOS driver
warnings.filterwarnings(    action="ignore",
                            module="scipy",
                            message="^internal gelsd")

# Column names for Data in imported CSV
CGM_COL_NAME = 'quantity'
INSULIN_COL_NAME = 'units'
MEAL_COL_NAME = 'meal'

# File path to read CGM data from
F_NAME = 'data/minimal_tracking_data.2019-05-32.csv'

# File path to store Latex tables to
TEX_PATH = './latex_outputs/'

# File path to store hyper parameter serach results
HP_PKL_PATH = './parameter_searches/search_%s.pkl'%\
                (time.strftime("%Y-%m-%d-%H:%M"))


def main():

    # Fetch data
    df = pd.read_csv(F_NAME)

    # Print data stats
    #DataStats = DataStatistics(df)
    #print(DataStats.summary())
    # DataStats.to_tex(TEX_PATH)

    # Structure Data from DataFrame
    USR_IDS = df['user_id'].unique()
    dfs = df.groupby('user_id')
    dfs = [dfs.get_group(u_id) for u_id in USR_IDS]
    data = [np.hstack([
            df[INSULIN_COL_NAME].values.reshape(-1,1).astype(float),
            df[CGM_COL_NAME].values.reshape(-1,1).astype(float),
            df[MEAL_COL_NAME].values.reshape(-1,1)]).astype(float)
            for df in dfs]

    # Pick single user for testing
    X, y = data[0][:,:2], data[0][:,2]

    # Run Model
    M = MealPredictionModel()
    M.fit(X)
    M.score(y, verbose=True)
    M.store_model(path='./results/m-test.pkl')


    # # Aggregate data over users
    # X_agg = [usr_df[['units', 'quantity']].values[:N,:] for usr_df in dfs]
    # y_agg = [usr_df[['meal']].values[:N] for usr_df in dfs]
    # # print_all_data_shapes(X_agg, y_agg)
    #
    # # GridSearch Params
    # param_grid = {
    #         'horizon': [30, 45],#[30, 45],
    #         'g': np.linspace(.0001, 5, 10), #np.random.uniform(1./100, 1., 3),
    #         'h': np.linspace(.0001, 5, 10), #np.random.uniform(1./100, 1., 3)
    #         'meal_threshold': np.linspace(2,10,2)
    # }
    #
    # grid_search = GridSearchCV(
    #         estimator=AggregatedMealModel(),
    #         param_grid=param_grid,
    #         iid=False,
    #         cv=DisableCV(),
    #         refit=False,
    #         verbose=2)
    # grid_search.fit(X_agg, y_agg)
    #
    # print('Best params:', grid_search.best_params_)
    # print('Best score:', grid_search.best_score_)
    #
    # # Pickle CV results
    # with open(HP_PKL_PATH, 'wb') as handle:
    #     pickle.dump(grid_search, handle, protocol=pickle.HIGHEST_PROTOCOL)
    #
    # # Run with best params
    # bestModel = AggregatedMealModel(**grid_search.best_params_)
    # bestModel.fit(X_agg, y_agg)
    # bestModel.score(X_agg, y_agg, mode='mean', verbose=1)
    #
    # print('*** EOF ***')


if __name__ == '__main__':
    main()
