import numpy as np
import pandas as pd
from sklearn.covariance import LedoitWolf
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.preprocessing import LabelEncoder


class Crossnobis:
    """
    Class to handle calculation of RDMs using crossvalidated mahalanobis (crossnobis) distance.
    RSA analogue to classifier object.
    Currently, just used as a place to store functions.


    TODO: Maybe offer more distance metric calculations?

    """

    def __init__(self):
        pass

    def _mean_by_condition(self, X, conds):
        """
        computes the average of each condition in X, ordered by conds
        returns a n_conditions x n_channels array

        Args:
            X (np.ndarray, shape (n_trials,n_channels)): Data to be averaged
            conds (np.ndarray): Condition labels

        Returns:
            avs (np.ndarray, shape (n_conditions, n_channels)): Average value for each condition
        """
        avs = np.zeros((len(np.unique(conds)), *X.shape[1:]))
        for cond in sorted(np.unique(conds)):
            X_cond = X[conds == cond]
            avs[cond] = X_cond.mean(axis=0)
        return avs

    def _means_and_prec(self, X, conds):
        """
        Returns condition averages and demeaned inverse covariance (precision matrix)
        Covariance is regularized by ledoit-wolf procedure

        Args:
            X (np.ndarray, shape (n_trials,n_channels)): Data to be averaged
            conds (np.ndarray): Condition labels

        Returns:
            cond_means (np.ndarray, shape (n_conditions, n_channels)): Average value for each condition
            inv_cov (np.ndarray, shape (n_channels, n_channels)): Inverse covariance matrix
        """
        cond_means = self._mean_by_condition(X, conds) # get condition averages
        cond_means_for_each_trial = cond_means[conds] # get a trials x channels array of mean values
        X_demean = X - cond_means_for_each_trial  # demean
        inv_cov = LedoitWolf(assume_centered=True).fit(X_demean).precision_

        return cond_means, inv_cov

    def _calc_rdm_crossnobis_single(self, X_train, X_test, precision):
        """
        Calculates RDM using LDC using means from x and y, and covariance
        Largely taken from https://github.com/rsagroup/rsatoolbox/blob/main/src/rsatoolbox/rdm/calc.py#L469
        Updated to return the signed square root of the RDM because
        LDC is an estimator of the squared mahalonobis distance

        Args:
            X_train (np.ndarray, shape (n_conditions, n_channels)): Condition averages for training data (first measure)
            meas2 (np.ndarray, shape (n_conditions, n_channels)): Condition averages for testing data (second measure)
            noise (np.ndarray, shape (n_channels, n_channels)): Precision (inverse covariance) matrix

        Returns:
            rdm (np.ndarray, shape (n_conditions, n_conditions)): RDM
        """
        kernel = X_train @ precision @ X_test.T
        rdm = np.expand_dims(np.diag(kernel), 0) + np.expand_dims(np.diag(kernel), 1) - kernel - kernel.T
        return np.sign(rdm) * np.sqrt(np.abs(rdm))

    def crossnobis_single(self, X_train, X_test, y_train, y_test):
        """
        Wrapper function to calculate crossnobis RDM over a single fold
        Uses condition means from both train and test, but only uses the training
        examples to compute the noise covariance/precision matrix. You may have another
        preference, but I did it this way to avoid train-test leakage.

        Args:
            X_train (np.ndarray, shape (n_trials, n_channels)): Training data
            X_test (np.ndarray, shape (n_trials, n_channels)): Testing data
            y_train (np.ndarray): Training labels
            y_test (np.ndarray): Testing labels
        
        Returns:
            rdm (np.ndarray, shape (n_conditions, n_conditions)): RDM
        """
        means_train, noise_train = self._means_and_prec(X_train, y_train)
        means_test = self._mean_by_condition(X_test, y_test)
        rdm = self._calc_rdm_crossnobis_single(means_train, means_test, noise_train)
        return rdm

    def crossnobis_across_time(self, X_train, X_test, y_train, y_test):
        """
        Wrapper function to calculate crossnobis RDM across timepoints in dataset. Use this as your main function

        Args:
            X_train (np.ndarray, shape (n_trials, n_channels, n_times)): Training data
            X_test (np.ndarray, shape (n_trials, n_channels, n_times)): Testing data
            y_train (np.ndarray): Training labels
            y_test (np.ndarray): Testing labels

        Returns:
            rdm (np.ndarray, shape (n_times, n_conditions, n_conditions)): RDM over time
        """
        ntimes = X_train.shape[2]

        rdm = np.stack(
            [
                self.crossnobis_single(X_train[:, :, itime], X_test[:, :, itime], y_train, y_test)
                for itime in range(ntimes)
            ]
        )
        return rdm
