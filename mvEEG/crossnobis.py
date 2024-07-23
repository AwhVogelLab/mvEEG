import numpy as np
import pandas as pd
from sklearn.covariance import LedoitWolf
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.preprocessing import LabelEncoder


class Crossnobis:
    def __init__(self):
        pass

    def _mean_by_condition(self, X, conds):
        '''
        computes the average of each condition in X, ordered by conds
        returns a n_conditions x n_channels array
        '''
        avs = np.zeros((len(np.unique(conds)), *X.shape[1:]))
        for cond in sorted(np.unique(conds)):
            X_cond = X[conds == cond]
            avs[cond] = X_cond.mean(axis=0)
        return avs

    def _means_and_prec(self, X, conds):
        '''
        Returns condition averages and demeaned inverse covariance
        Covariance is regularized by ledoit-wolf procedure
        '''
        cond_means = self._mean_by_condition(X, conds)
        cond_means_for_each_trial = cond_means[conds]
        X_demean = X - cond_means_for_each_trial  # demean

        return cond_means, LedoitWolf(assume_centered=True).fit(X_demean).precision_

    def _calc_rdm_crossnobis_single(self, meas1, meas2, noise):
        '''
        Calculates RDM using crossnobis distance using means from x and y, and covariance
        Largely taken from https://github.com/rsagroup/rsatoolbox/blob/main/src/rsatoolbox/rdm/calc.py#L429
        Updated to return the signed square root of the RDM because
        LDC is an estimator of the squared mahalonobis distance
        '''
        kernel = meas1 @ noise @ meas2.T
        rdm = np.expand_dims(np.diag(kernel), 0) + \
            np.expand_dims(np.diag(kernel), 1) - kernel - kernel.T
        return np.sign(rdm) * np.sqrt(np.abs(rdm))

    def crossnobis_single(self, X_train, X_test, y_train, y_test):
        '''
        Uses condition means from both train and test, but only uses the training
        examples to compute the noise covariance/precision matrix. You may have another
        preference, but I did it this way to avoid train-test leakage. 
        '''
        means_train, noise_train = self._means_and_prec(X_train, y_train)
        means_test = self._mean_by_condition(X_test, y_test)
        rdm = self._calc_rdm_crossnobis_single(
            means_train, means_test, noise_train)
        return rdm
    
    def crossnobis_across_time(self, X_train, X_test, y_train, y_test):
        ntimes = X_train.shape[2]

        rdm = np.stack([self.crossnobis_single(
                X_train[:, :, itime], X_test[:, :, itime], y_train, y_test)
                for itime in range(ntimes)])
        return rdm