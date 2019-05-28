import inspect

import numpy as np

from scipy.interpolate import pchip_interpolate
from scipy.signal import savgol_filter


class PreProcess(object):
    """Preprocessing pipeline for CGM, bolus and meal dataFrames.
    Preprocessing includes:
        - Resampling, frequency given by resample_period argument (default 1Min)
        - Conversion, transform insulin dosing from Units to mg.
        - Smoothing, by Savgol filter.
    """


    def __init__(self, horizon=45, data_frequency=5, do_insulin_to_mg=True,
                       do_savgol_smoothing=True, savgol_poly=1, savgol_len=15,
                       interpolation_method='pchip'):

        # Short hand for assigning all args to attributes with same name
        args, _, _, values = inspect.getargvalues(inspect.currentframe())
        values.pop("self")
        for arg, val in values.items():
            setattr(self, arg, val)


    def apply(self, df):
        """Apply PreProcessing pipeline to df and return updated df"""

        # Pull out insulin and cgm arrays
        insulin_inputs = df['insulinDeliverySampleEvent'].values
        cgm_inputs = df['bloodGlucoseSampleEvent'].values

        # Chunk arrays into sliding window sized chunks
        insulin_inputs = self.split_into_horizon_chunks(insulin_inputs)
        cgm_inputs = self.split_into_horizon_chunks(cgm_inputs)

        # Upsample convert (insulin) and smoothen (cgm)
        insulin_inputs = self.preprocess_insulin(insulin_inputs)
        cgm_inputs = self.preprocess_cgm(cgm_inputs)

        return (insulin_inputs, cgm_inputs)


    def split_into_horizon_chunks(self, arr):
        """Chunk input array to represent sliding window of size=horizon"""
        window = self.horizon//self.data_frequency
        arr = [arr[i:i+window+1] for i in range(0, len(arr)-window)]
        return arr


    def preprocess_insulin(self, insulin_inputs):
        """Process insulin by upsampling and converting to mg"""
        insulin_inputs = [self.upsample_insulin(chunk)
                         for chunk in insulin_inputs]

        if self.do_insulin_to_mg:
            insulin_inputs = [self.convert_units_to_mg(chunk)
                             for chunk in insulin_inputs]

        return insulin_inputs


    def upsample_insulin(self, insulin_chunk):
        """Upsample to 1 minute resolution and cast as float."""

        # Set all idxs where zero padding should be place
        pad_idxs = [i for i in range(1,len(insulin_chunk))]
        pad_idxs = np.repeat(pad_idxs, self.data_frequency-1)
        # Insert zeros to upsample data resolution
        insulin_chunk = np.insert(insulin_chunk, pad_idxs, 0, axis=0)

        return insulin_chunk.astype(float)


    def convert_units_to_mg(self, units):
        """Convert administred insulin (boluses) form Units to milligram (mg)"""

        return units*0.347 #0.0347


    def preprocess_cgm(self, cgm_inputs):
        """Docstring..."""

        cgm_inputs = [self.upsample_interpolate(chunk) for chunk in cgm_inputs]
        cgm_inputs = [self.savgol_smoothing(chunk) for chunk in cgm_inputs]

        return cgm_inputs

    def upsample_interpolate(self, cgm_chunk):
        """Docstring..."""

        x_axis = np.arange(0,self.horizon+1,self.data_frequency)
        return pchip_interpolate(x_axis, cgm_chunk, np.arange(self.horizon+1))


    def savgol_smoothing(self, signal):
        """Smoothen the CGM data by using a Savgol filter, parameters and
        previous results on CGM smoothing can be found in:
            - https://pubs.acs.org/doi/abs/10.1021/ie3034015
        """

        return savgol_filter(signal,
                             window_length=self.savgol_len,
                             polyorder=self.savgol_poly,
                             mode='interp')
