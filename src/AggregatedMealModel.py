import inspect

import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin

from MealPredictionModel import MealPredictionModel


class AggregatedMealModel(BaseEstimator, ClassifierMixin):
    """Aggregate predictive models for various patients.

    This class allows to aggregate results over different patients given one
    set of Model parameters. This in turn allows to run a parameter search
    to find best parameters for all models.
    """

    def __init__    (self, horizon=45, data_frequency=5, x0=0, dx=0, g=.6,
                    h=.01, dt=1., meal_duration=45, savgol_poly=1,
                    savgol_len=15, calibration_time=3, std_mult=2,
                    prediction_period=60):

        """Initialize aggregation class. All keyword arguments are passed
        along to each patient's individual MealPrediction Model.

        Keyword arguments:

        horizon -- Prediction horizon for glucose values.
        data_frequency -- The sample period of data that's fed to the model.
        x0 -- Is the initial guess for glucose in Gut compartment.
        dx -- Is the initial change rate for glucose rate of change.
        g -- Is the g-h filter's g scale factor
        h -- Is the g-h filter's h scale factor
        dt -- Is the length of the time step
        meal_duration -- Duration for meal flag (default 15 min).
        savgol_poly -- DoF for Savgol smoothing filter.
        savgol_len -- Length of Savgol smoothing window.
        calibration_time -- Gh filter will produce some ringing, allow cal time.
        std_mult -- Meal prediction are based on the estimate glucose
                consumption offset from mean in no. of std deviations.
        prediction_period -- Time span after a meal where a prediction counts
                as a true positive.
        """

        # Short hand for assigning all args to attributes with same name
        args, _, _, values = inspect.getargvalues(inspect.currentframe())
        values.pop("self")
        for arg, val in values.items():
            setattr(self, arg, val)


    def fit(self, X, y=None):
        """Fit one MealPrediction model per patient.

        Keyword arguments:
        X -- Array with each element representing each patient's data of CGM
             and bolus data.
        y -- None, not used but required by Sklearn for interface with CVsearch.
        """

        self.n_users_ = len(X)
        self.models_ = [0]*self.n_users_

        for it, usr_X in enumerate(X):

            self.models_[it] = MealPredictionModel(
                    self.horizon, self.x0, self.dx, self.g, self.h, self.dt,
                    self.meal_duration, self.meal_threshold)
            self.models_[it].fit(usr_X)

        return self


    def score(self, X, y=None, mode='TPR', verbose=0):
        """Score each individual MealPrediction model and return mean.

        Keyword arguments:
        X -- Vectors to fit model on.
        y -- Target vecor for all users.
        mode -- What type of score we're interested in.
        verbose -- Print score of each individual model (1) or not (0).
        """

        self.usr_scores_ = [0]*self.n_users_

        for it, usr_y in enumerate(y):
            self.usr_scores_[it] = self.models_[it].score(y=usr_y,
                                                          mode=mode,
                                                          verbose=verbose)

        self.usr_scores_ = np.array(self.usr_scores_)
        return self.usr_scores_.mean()
