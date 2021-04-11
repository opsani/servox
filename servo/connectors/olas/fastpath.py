# This Module is a simplified online prediction model that predicts immediate
# future traffic based on traffic history of short term and high frequency.

import numpy as np
import pandas as pd
import pygam


class Fastpath(object):
    '''
    Online prediction model that predicts immediate future traffic.

    param:
        - data: history data used to instantiate Fastpath model
        - foresight: in mins, denotes how much in advance the prediction will be,
                     and also what frequency used to downsample raw data.
                     Should be larger than the cadence of received measurements.
        - window_size: window size used to smooth resampled time series
    '''

    def __init__(self, data, foresight=1, window_size=5):
        self.data_lake = self.build_data_lake(data)
        self.window_size = int(window_size)
        self.foresight = int(foresight)

        self.resample()
        # window_size indicats half width of intended sliding window, thus 4 in below
        # condition is chosen to guarantee at least half of resampled data could be
        # applied with hampel_filter. If condition failed, normally it is due to
        # insufficient data quantity when controller has not monitored application
        # long enough, but it may also be caused by poorly chosen window_size that
        # requires significant amount of data (the bigger the window_size, the more
        # data needed). 5 is an empirical sound value.
        if len(self.traffic) < 4 * self.window_size:
            raise ValueError(
                "Fastpath Argument Error: insufficient traffic data. "
                f"Need data of at least {4*self.window_size*self.foresight} mins duration.")

    def build_data_lake(self, data):
        '''
        Transform received data to pandas dataframe and index it with timestamps.
        '''
        # Instantiate a dataframe
        temp = pd.DataFrame(data, columns=['ts', 'total'])
        # Change index
        temp.set_index('ts', inplace=True)
        # Epoch time to human readable time
        temp.index = pd.to_datetime(temp.index, unit='s')

        return temp

    def resample(self):
        '''
        Downsample recived data to mean on specified resampling frequency.
        '''
        # Downsampled received raw data
        resampled = self.data_lake.resample(str(self.foresight) + 'T', closed='right', label='right').mean()
        # Interpolate missing values
        resampled['total'] = resampled['total'].interpolate(method='polynomial', order=2)
        # Extract processed traffic values
        self.traffic = resampled['total'].values

    def stream(self, datum):
        '''
        Append newest streaming datum to current data lake and discard oldest datum.
        '''
        # Cast newest datum to pandas dataframe
        temp = self.build_data_lake(datum)
        # Append new datum
        self.data_lake = self.data_lake.append(temp)
        # Discard oldest datum
        self.data_lake.drop(self.data_lake.index[0], inplace=True)

    def predict(self, datum=None):
        '''
        Carry out online prediction and return predicted numeric value.
        '''
        if datum is not None:
            # Incorporate newest streaming datum
            self.stream(datum)
            # Downsample
            self.resample()

        # Cache to store predictions
        self.momentums = []

        # Make predictions in a rolling manner
        for i in range(len(self.traffic) - self.window_size - 1, len(self.traffic)):

            if i < 3 * self.window_size:
                self.momentums.append(self.traffic[i])
                continue

            prop_traffic = self.traffic[i - 3 * self.window_size:i]

            # apply Hampel filter to identify and replace outliers
            new_traffic = self.hampel_filter(prop_traffic)

            # apply Generalized additive model
            model = pygam.LinearGAM(pygam.s(0, n_splines=10, basis='ps', spline_order=3))\
                         .gridsearch(np.arange(0, len(new_traffic))[:, None],
                                     np.array(new_traffic) * 1.05,
                                     progress=False, objective='AICc')
            momentum = model.predict(np.arange(0, len(new_traffic) + 1)).flatten()[-1]
            self.momentums.append(momentum)

        # apply attenuate fiter to lessen traffic plunge and traffic soar
        prediction = self.attenuate_filter(self.momentums)

        return prediction

    def hampel_filter(self, signals, n_sigmas=3):
        '''
        Hampel Filter
        '''
        n = len(signals)
        new_series = signals.copy()
        k = 1.4826  # scale factor for Gaussian distribution

        for i in range((self.window_size), (n - self.window_size)):
            x0 = np.median(signals[(i - self.window_size):(i + self.window_size)])
            S0 = k * np.median(np.abs(signals[(i - self.window_size):(i + self.window_size)] - x0))
            if np.abs(signals[i] - x0) > n_sigmas * S0:
                new_series[i] = x0

        return new_series

    def attenuate_filter(self, signals, n_sigmas=3):
        '''
        Customized filter
        '''
        diff = np.array(signals[1:]) - np.array(signals[:-1])
        k = 1.4826  # scale factor for Gaussian distribution

        d0 = np.median(np.abs(diff))
        S0 = k * np.median(np.abs(np.abs(diff) - d0))

        if diff[-1] < 0 and (abs(diff[-1]) - d0) > 3 * S0:
            return signals[-1] + 3 * S0
        elif diff[-1] > 0 and (abs(diff[-1]) - d0) > 3 * S0:
            return signals[-1] - 3 * S0
        else:
            return signals[-1]
