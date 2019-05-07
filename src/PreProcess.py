import inspect

import numpy as np

from scipy.signal import savgol_filter


class PreProcess(object):
    """Preprocessing pipeline for CGM, bolus and meal dataFrames.
    Preprocessing includes:
        - Resampling, frequency given by resample_period argument (default 1Min)
        - Conversion, transform insulin dosing from Units to mg.
        - Smoothing, by Savgol filter.
    """


    def __init__(self, resample_period='1T', do_insulin_to_mg=True,
                       do_savgol_smoothing=True, savgol_poly=1, savgol_len=15,
                       interpolation_method='pchip'):

        # Short hand for assigning all args to attributes with same name
        args, _, _, values = inspect.getargvalues(inspect.currentframe())
        values.pop("self")
        for arg, val in values.items():
            setattr(self, arg, val)

    def apply(self, df):
        """Apply PreProcessing pipeline to df and return updated df"""

        # Process Insulin and Glucose values separately
        insulin_df, glucose_df = self.process_insulin(df),\
                                self.process_cgm_values(df)
        if self.do_insulin_to_mg:
            insulin_df['units'] = insulin_df['units']\
                                  .apply(lambda x: self.convert_units_to_mg(x))
        X = insulin_df.join(glucose_df)

        # Extract logged meals and merge into one dataFrame
        y = df[df['meal'] == 1][['meal']]
        df = X.join(y)

        # Smoothen CGM data
        df.replace(np.nan, 0., inplace=True)
        if self.do_savgol_smoothing:
            df['quantity'] = self.savgol_smoothing(df['quantity'])

        return df


    def process_insulin(self, df):
        """Pick data where bolusing is present, resample and cast as float.
        Return processed dataFrame.
        """

        df = df[df['insulinDeliverySampleEvent'] == 1][['units']]
        df = df.replace(np.nan, 0.)
        df['units'] = df['units'].astype(float)
        df = df.resample('1T').sum()

        return df


    def process_cgm_values(self, df):
        """Pick data where CGM values is present, resample and remove duplicate
        measurements which occurs infrequently on multiple CGM sensors.
        Interpolate to fill missing values after resampling.
        """

        df = df[df['bloodGlucoseSampleEvent'] == 1][['quantity']]
        df = df[~df.index.duplicated(keep='first')]
        df = df.resample('1T').interpolate(method=self.interpolation_method)

        return df


    def savgol_smoothing(self, signal):
        """Smoothen the CGM data by using a Savgol filter, parameters and
        previous results on CGM smoothing can be found in:
            - https://pubs.acs.org/doi/abs/10.1021/ie3034015
        """

        return savgol_filter(signal,
                             window_length=self.savgol_len,
                             polyorder=self.savgol_poly)


    def convert_units_to_mg(self, units):
        """Convert administred insulin (boluses) form Units to milligram (mg)"""

        return units*0.347 #0.0347
