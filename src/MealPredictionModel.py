import inspect

import numpy as np
import pandas as pd
import pickle

from scipy.interpolate import pchip_interpolate
from scipy.signal import savgol_filter
from sklearn.base import BaseEstimator, ClassifierMixin

from MinimalBergmanModel import BergmanModel
from G_h_filter import G_h_filter


class MealPredictionModel(BaseEstimator, ClassifierMixin):
    """Compartment model for predicting meals from CGM data.

    The physiological model, which describes the glucoregulatory network,
    incorporates a twocompartment glucose subsystem, an insulin subsystem,
    and a three-compartment insulin action subsystem.

    The model is designed to take a Bolus injection and CGM measurements
    time series data. Among with a carbohydrate input estimation future
    glocode values are predicted. Prediction and cgm-meaasurement residuals
    are provided with carbohydrate estimate to an alpha-beta filter to refine
    the carbohydrate intake estimation.

    When a carbohydrate estimation significantly higher than the average
    (one standard deviation away) is registred a the model predicts a meal
    at the goven time step.
    """

    def __init__    (self, horizon=45, data_frequency=5, x0=0, dx=0, g=.6,
                    h=.01, dt=1., meal_duration=60, meal_threshold=10,
                    savgol_poly=1, savgol_len=15):

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


    def split_into_horizon_chunks(self, arr):
        """Chunk input array to represent sliding window of size=horizon"""

        window = self.horizon//self.data_frequency
        for i in range(0, len(arr)-window):
            yield arr[i:i+window+1]


    def fit(self, X, y=None):
        """Predict meals for a time period N given insulin and cgm measurements.

        Keyword arguemnts:
        X -- Numpy array with shape (N,2).
            Col 0 -- Time series of length N with Bolus data.
            Col 1 -- Time series of length N with CGM data.
        y -- Not used, argument required by sklearn.
        """

        # Initialize new variables
        N = X.shape[0]
        self.bolus_inputs_ = X[:,0]
        self.cgm_inputs_ = X[:,1]

        self.carb_ests_ = np.zeros(N)
        self.carb_mean_ = np.zeros(N)
        self.carb_stds_ = np.zeros(N)
        self.cgm_preds_ = np.zeros((N,self.horizon+1))

        self.meal_preds_ = np.zeros(N)
        self.meal_flag_ = False
        self.meal_count_ = self.meal_duration

        self.G_h_filter_ = G_h_filter(self.x0, self.dx, self.g, self.h, self.dt)

        # Initialize compartments
        self._init_compartment_vars(N)

        # Prepare sliding window chunks
        bolus_generator = self.split_into_horizon_chunks(self.bolus_inputs_)
        cgm_generator = self.split_into_horizon_chunks(self.cgm_inputs_)

        for it, (bolus_vector, cgm_vector) in enumerate(zip(bolus_generator,
                                                            cgm_generator)):
            # Process inputs
            bolus_vector = self.process_insulin(bolus_vector)
            cgm_vector = self.process_cgm(cgm_vector)

            # Naive carb estimate
            carb_est = self.G_h_filter_.predict()

            # Compartment Model CGM Predictions
            cgm_prediction = self.CompartmentModel.predict_n(n=self.horizon+1,
                                                             bolus=bolus_vector,
                                                             food=carb_est)
            self.cgm_preds_[it] = cgm_prediction
            # Residual based carb estimate update
            carb_est = self.G_h_filter_.update(cgm_vector,
                                               cgm_prediction)

            # Save new estimate and update Compartment Model
            self.carb_ests_[it-1] = carb_est
            self.CompartmentModel.update_compartments(bolus_vector[0],
                                                      carb_est)

            # Save compartmen values
            self._add_compartment_vars(it)

            # Check for Meal detection
            self._check_meal_alarm(carb_est, self.carb_ests_, it)

        return self


    def process_insulin(self, bolus_vector):
        """Process insulin by upsampling to 1 minute resolution
        and convert Units to to mg.
        """

        bolus_vector = self.upsample_insulin(bolus_vector)
        bolus_vector = self.convert_units_to_mg(bolus_vector)

        return bolus_vector


    def upsample_insulin(self, bolus_vector):
        """Upsample to 1 minute resolution and cast as floats."""

        # Set all indices where zero padding should be placed
        pad_idxs = np.arange(1,len(bolus_vector))
        pad_idxs = np.repeat(pad_idxs, self.data_frequency-1)

        # Insert zeros to upsample data resolution
        bolus_vector = np.insert(bolus_vector, pad_idxs, 0, axis=0)

        return bolus_vector.astype(float)


    def convert_units_to_mg(self, units):
        """Convert administred insulin (boluses) from Units to mg."""

        return units*0.347


    def process_cgm(self, cgm_vector):
        """Docstring..."""

        cgm_vector = self.upsample_interpolate(cgm_vector)
        cgm_vector = self.savgol_smoothing(cgm_vector)

        return cgm_vector


    def upsample_interpolate(self, cgm_vector):
        """Docstring..."""

        x_axis = np.arange(0,self.horizon+1,self.data_frequency)
        return pchip_interpolate(x_axis, cgm_vector, np.arange(self.horizon+1))


    def savgol_smoothing(self, signal):
        """Smoothen the CGM data by using a Savgol filter, parameters and
        previous results on CGM smoothing can be found in:
            - https://pubs.acs.org/doi/abs/10.1021/ie3034015
        """

        return savgol_filter(signal,
                             window_length=self.savgol_len,
                             polyorder=self.savgol_poly,
                             mode='interp')


    def _check_meal_alarm(self, curr_est, carb_estimates, it):
        """Return meal detection predictions based of estimated glucose values

        Keyword arguemnts:
        cgm_series -- Measured CGM values.
        gt_series -- Cgm predictions from g-h filter over Bergman model.
        """

        # Allow for 3 hours (60 min * 3) calibration time
        if it < 60*3:
            return

        # Find mean and std of carb estimates so far
        c_mean = carb_estimates.mean()
        c_std = carb_estimates.std()

        # Store values
        self.carb_mean_[it-1] = c_mean
        self.carb_stds_[it-1] = c_std

        # Raise meal detected when threshold is exceeded - keep meal flag up to avoid dupes
        if curr_est >= c_mean + c_std and self.meal_flag_ is False:
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


    def score(self, y, mode='PPV', verbose=0):
        """Given a vector of ground truth, score the prediction made.

        Keyword arguemnts:
        y -- Binary array with 1 marking timesteps where meal were logged.
        mode -- Select score type.
        verbose -- Print the final score (1) or not (0).
        """

        # Reshape and clip labels according to horizon
        y = y.reshape(-1)
        #y = y[:-self.horizon]
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

            if it < 3*60:
                # Allow calibration time
                continue

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
        if prediction_offsets.shape[0] != 0:
            mean_std = (prediction_offsets.mean(), prediction_offsets.std())
        else:
            mean_std = (300, 300)

        self.scores_ = {
            # Detection offsets and standard deviation (mean):
            'mean': mean_std,
            # Precision or Positive Predictive Value (PPV):
            'PPV': TP/(TP+FP),
            # False Discovery Rate (FDR):
            'FDR': 1 - (TP/(TP+FP))
        }
        if verbose:
            print(self.scores_)

        return self.scores_[mode]


    def store_model(self, path):

        df = pd.DataFrame.from_records({
            'bolus_inputs_': self.bolus_inputs_,
            'cgm_inputs_': self.cgm_inputs_,
            # 'cgm_preds_': self.cgm_preds_,
            'carb_ests_': self.carb_ests_,
            'carb_mean_': self.carb_mean_,
            'carb_stds_': self.carb_stds_,
            'meal_preds_': self.meal_preds_,
            'y_meals': self.y_meals,
            's1': self.s1,
            's2': self.s2,
            'gt': self.gt,
            'mt': self.mt,
            'It': self.It,
            'Xt': self.Xt,
            'Gt': self.Gt
            # 'scores_': self.scores_
        })

        df.to_pickle(path)

        with open('./results/m-test-1-pred.pkl', 'wb') as f:
            pickle.dump(self.cgm_preds_, f)
