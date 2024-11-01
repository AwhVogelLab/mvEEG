import numpy as np
from sklearn.covariance import LedoitWolf
from sklearn.preprocessing import LabelEncoder
from rsatoolbox.rdm.calc import _calc_rdm_crossnobis_single


class Crossnobis:
    """
    Class to handle calculation of RDMs using crossvalidated mahalanobis (crossnobis) distance.
    RSA analogue to classifier object.
    Currently, just used as a place to store functions.


    TODO: Maybe offer more distance metric calculations?

    """

    def __init__(self, labels):
        self.labels = labels
        self.n_labels = len(labels)
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
        cond_means = self._mean_by_condition(X, conds)  # get condition averages
        cond_means_for_each_trial = cond_means[conds]  # get a trials x channels array of mean values
        X_demean = X - cond_means_for_each_trial  # demean
        inv_cov = LedoitWolf(assume_centered=True).fit(X_demean).precision_

        return cond_means, inv_cov

    def crossnobis(self, X_train, X_test, y_train, y_test):
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
        rdm = _calc_rdm_crossnobis_single(means_train, means_test, noise_train)
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
            [self.crossnobis(X_train[:, :, itime], X_test[:, :, itime], y_train, y_test) for itime in range(ntimes)]
        )
        return rdm


def temporally_generalize(self, X_train, X_test, y_train, y_test):
    ntimes = X_train.shape[2]
    rdms = np.full((self.n_labels, self.n_labels, ntimes, ntimes), np.nan)

    for itime in range(ntimes):  # train times
        means_i, noise_i = self._means_and_prec(X_train[:, :, itime], y_train)
        for jtime in range(ntimes):  # test times
            means_j = self._mean_by_condition(X_test[:, :, jtime], y_test)
            rdms[:, itime, jtime] = _calc_rdm_crossnobis_single(means_i, means_j, noise_i)

    return rdms
