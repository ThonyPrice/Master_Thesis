import inspect

import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin

from MinimalBergmanModel import BergmanModel
from G_h_filter import G_h_filter


class MealPredictionModel(BaseEstimator, ClassifierMixin):
    """Compartment model for predicting meals from CGM data.

    The physiological model, which describes the glucoregulatory network,
    incorporates a twocompartment glucose subsystem, an insulin subsystem,
    and a three-compartment insulin action subsystem.

    The model is designed to and Bolus injection time series data
    and a glucose estimation obtained through a g-h filter to predict
    future CGM values.

    The residuals between predicted CGM and measured data points
    indicates a large unaccounted meal is introduced into the system and
    the final goal of this model is to detect said meals.
    """

    def __init__    (self, horizon=40, x0=0, dx=0, g=.6, h=.01, dt=1.,
                    meal_duration=30, meal_threshold=10):
        """Initialize Prediction class

        Keywork arguments:
        horizon -- How many timesteps to try predict glucose ahead of.
        x0 -- Is the initial guess for glucose in Gut compartment.
        dx -- Is the initial change rate for glucose rate of change.
        g -- Is the g-h’s g scale factor
        h -- Is the g-h’s h scale factor
        dt -- Is the length of the time step

        meal_threshold -- Gradient threshold values to trigger meal on.
        meal_duration -- Duration for meal flag (default 15 min).
        """

        self.CompartmentModel = BergmanModel()
        self.s1 = np.array([])
        self.s2 = np.array([])
        self.gt = np.array([])
        self.mt = np.array([])
        self.It = np.array([])
        self.Xt = np.array([])
        self.Gt = np.array([])

        args, _, _, values = inspect.getargvalues(inspect.currentframe())
        values.pop("self")

        for arg, val in values.items():
            setattr(self, arg, val)


    def fit(self, X, y=None):
        """Performs g-h filter on 1 state variable with a fixed g and h.

        Keyword arguemnts:
        X -- Numpy array with shape (N,2).
            Column 0 -- Time series of length N with Bolus data.
            Column 1 -- Time series of length N with CGM data.
        y -- Should not be provided to, argument required by sklearn.
        """

        # Prep data
        self.cgm_series_ = X[:-self.horizon,1]
        self.cgm_targets_ = self._shift_horizon(X[:,1], self.horizon)
        self.bolus_series_ = X[:-self.horizon,0]

        # Initialize new variables
        N = X.shape[0] - self.horizon
        self.rss_ = np.zeros(N)
        self.meal_preds_ = np.zeros(N)
        self.meal_flag_ = False
        self.meal_count_ = self.meal_duration
        self.G_h_filter_ = G_h_filter(self.x0, self.dx, self.g, self.h, self.dt)
        self._init_compartment_vars(N)

        for it, cgm in enumerate(self.cgm_series_):

            bolus_vector = self._verify_bolus_vector(self.bolus_series_,
                                                    self.horizon,
                                                    it)
            glucose_estimate = self.G_h_filter_.predict()
            cgm_prediction = self.CompartmentModel.predict_n(n=self.horizon,
                                                             bolus=bolus_vector,
                                                             food=glucose_estimate)
            glucose_est = self.G_h_filter_.update(self.cgm_targets_[it],
                                                  cgm_prediction)
            self.CompartmentModel.update_compartments(self.bolus_series_[it],
                                                      glucose_est)

            self._add_compartment_vars(it)
            self._check_meal_alarm(self.cgm_series_[:it], self.Gt[:it], it)

        return self


    def _check_meal_alarm(self, cgm_series, gt_series, it):
        """Return meal detection predictions based of estimated glucose values

        Keyword arguemnts:
        cgm_series -- Measured CGM values.
        gt_series -- Cgm predictions from g-h filter over Bergman model.
        """

        # Can't calculare gradient of RSS on a single value
        if it < 3:
            return

        # Find series of RSS gradient at each time step
        self.rss_ = (cgm_series-gt_series)**2
        rss_grads = np.gradient(self.rss_)

        # Raise meal detected when threshold is exceeded - keep meal flag up to avoid dupes
        if rss_grads[it-1] > self.meal_threshold and self.meal_flag_ is False:
                self.meal_preds_[it-1] = 1
                self.meal_flag_ = True

        if self.meal_flag_ is True:
            self.meal_count_ = (self.meal_count_-1)%self.meal_duration
            self.meal_flag_ = True if self.meal_count_ > 0 else False


    def _init_compartment_vars(self, N):
        """Initialize arrays for all subcompartments

        Keyword arguments:
        N -- Length of array.
        """
        self.s1 = np.zeros(N)
        self.s2 = np.zeros(N)
        self.gt = np.zeros(N)
        self.mt = np.zeros(N)
        self.It = np.zeros(N)
        self.Xt = np.zeros(N)
        self.Gt = np.zeros(N)


    def _add_compartment_vars(self, i):
        """Fetch values of subcompartments at timestep i

        Keyword arguments:
        i -- Current timestep.
        """
        s1, s2 = self.CompartmentModel.insulin_system.get_variables()
        gt, mt = self.CompartmentModel.glucose_system.get_variables()
        It, Xt, Gt = self.CompartmentModel.get_variables()

        self.s1[i] = s1
        self.s2[i] = s2
        self.gt[i] = gt
        self.mt[i] = mt
        self.It[i] = It
        self.Xt[i] = Xt
        self.Gt[i] = Gt


    def _shift_horizon(self, cgm_series, horizon):
        """Shift CGM series so that index of prediction and target are the same
        Return a shifted and clipped series of prediction targets.

        Keyword arguments:
        cgm_series -- Measured CGM values.
        horizon -- Offset of prediction basis and target.
        """
        return np.roll(cgm_series, -horizon)[:-horizon]


    def _verify_bolus_vector(self, bolus_series, horizon, it):
        """This functions catches the edge case when the length of future
        bolus injections exceed the available data. Return available data
        padded with zeros to the expected length.

        Keyword arguments:
        bolus_series -- Bolus injections time series data.
        horizon -- Required length of bolus data after iterationstep it.
        it -- Iteration step.
        """

        bolus_vector = bolus_series[it:it+horizon]
        if len(bolus_vector) < horizon:
            bolus_vector = np.pad(  bolus_vector,
                                    (0,horizon-len(bolus_vector)),
                                    'constant')
        return bolus_vector


    def score(self, y, mode='PPV'):
        """Given a vector of ground truth, score the prediction made.

        Keyword arguemnts:
        y -- Binary array with 1 marking timesteps where meal were logged.
        mode -- Select score type.
        """

        # Reshape and clip labels according to horizon
        y = y.reshape(-1)
        y = y[:-self.horizon]
        self.y_meals = y

        # Scoring function for comparing logged- and predicted meals.
        false_positive_idxs = [] # Meal detected w/o one occurring.
        true_positive_idxs = [] # Meal detected in proximity to one occurring.
        false_negative_idxs = [] # Meal not detected when occurring
        true_negative_idxs = [] # Meal not detected when one did not occurr

        time_since_meal = 0
        meal_occurred_flag = False

        last_meal_idx = None
        last_prediction_idx = None
        prediction_offsets = []
        predicted_meals = self.meal_preds_

        for it, meal in enumerate(y):

            if meal == 1:
            # Meal occurred

                if meal_occurred_flag == True:
                    # This is a another meal happening w/o a prediction being made for the last one
                    false_negative_idxs.append(last_meal_idx)
                    last_meal_idx = it

                if meal_occurred_flag == False:
                    # Raise a meal occurred flag - We now expect a prediction
                    last_meal_idx = it
                    meal_occurred_flag = True

            if predicted_meals[it] == 1:
                # Meal predicted

                if meal_occurred_flag == True:
                    # A meal has occurred
                    prediction_offsets.append(it-last_meal_idx)
                    true_positive_idxs.append(it)
                    meal_occurred_flag = False

                if meal_occurred_flag == False:
                    # Predicted meal w/o one occurring
                    false_positive_idxs.append(it)


        FP = len(false_positive_idxs)
        TP = max(1, len(true_positive_idxs))
        FN = len(false_negative_idxs)

        prediction_offsets = np.array(prediction_offsets)
        if prediction_offsets.size != 0:
            mean_std = (prediction_offsets.mean(), prediction_offsets.std())
        else:
            mean_std = (120, 120)

        self.scores_ = {
            # Detection offsets and standard deviation (mean):
            'mean': mean_std,
            # Precision or Positive Predictive Value (PPV):
            'PPV': TP/(TP+FP),
            # False Discovery Rate (FDR):
            'FDR': 1 - (TP/(TP+FP))
        }

        return self.scores_[mode]
