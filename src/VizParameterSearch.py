import pickle
import time

from matplotlib import cm
from matplotlib import pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import pandas as pd


class VizparameterSearch(object):
    """Visualize Parameter Search."""


    def __init__(self):
        """Initialize data with a GridSeachCV score dict"""
        self.arg = None


    def save_plot(self, plt):
        """Save plot to plots folder"""

        f_name = './plots/plot_%s.png'%\
                 (time.strftime("%Y-%m-%d-%H:%M"))

        plt.savefig(f_name)


    def plot(self, f_name=None, x='g', y='h', z=None, horizon=45):
        """Surface plot to detail results of parameter seach

        Keyword arguemts:
        f_name -- File name to retreive gridSearch data from.
        horizon -- Pick one of the available prediction horizons.

        Partial source code from: https://stackoverflow.com/a/49606208
        """

        # Testing
        f_name = "./parameter_searches/search_2019-05-08-16:53.pkl"

        with open(f_name, 'rb') as file:
            file_data = pickle.load(file)

        cv_results = file_data.cv_results_
        scores_df = pd.DataFrame(cv_results)

        #scores_df = scores_df[scores_df.param_meal_threshold == 2]
        scores_df = scores_df[scores_df.param_meal_threshold == 10]
        #scores_df = scores_df[scores_df.param_horizon == 30]
        scores_df = scores_df[scores_df.param_horizon == 30]

        M = len(scores_df.param_g.unique())
        N = len(scores_df.param_h.unique())
        X = scores_df.param_g.values.reshape(M,N)
        Y = scores_df.param_h.values.reshape(M,N)
        Z = scores_df.split0_test_score.values.reshape(M,N)

        fig = plt.figure()
        ax = fig.gca(projection='3d')
        surf = ax.plot_surface(X, Y, Z, cmap=cm.jet, linewidth=0.1)
        fig.colorbar(surf, shrink=0.5, aspect=5)

        ax.set_xlabel('g param')
        ax.set_ylabel('h_param')
        ax.set_zlabel('Score')

        self.save_plot(plt)
        #plt.show()

obj = VizparameterSearch()
obj.plot()
