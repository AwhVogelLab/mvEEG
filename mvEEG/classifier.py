import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import confusion_matrix


class Classifier:
    """
    Class to run basic decoding algorithms on EEG data.

    Parameters:
        labels (list): List of integer labels to be used for decoding
        classifier (sklearn classifier), default LogisticRegression: Classifier to be used for decoding
        scaler (sklearn scaler), default StandardScaler: Scaler to be used for decoding
    """

    def __init__(self, labels,classifier=None, scaler=None):
        self.labels = labels
        self.n_labels = len(labels)
        if classifier is not None:
            self.classifier = classifier
        else:
            self.classifier = LogisticRegression()
        if scaler is not None:
            self.scaler = scaler
        else:
            self.scaler = StandardScaler()
        self.rng = np.random.default_rng()

    def _standardize(self, X_train, X_test):
        """
        Helper function to standardize data. 
        Standardizes test to mean and std of the training set

        Args:
            X_train (np.array): Training data
            X_test (np.array): Testing data

        """

        X_train = self.scaler.fit_transform(X_train)
        X_test = self.scaler.transform(X_test)
        return X_train, X_test

    def _decode(self, X_train, X_test, y_train, y_test):
        """
        Helper function to return decoding metrics on a single timepoint

        Args:
            X_train (np.ndarray): Training data
            X_test (np.ndarray): Testing data
            y_train (np.ndarray): Training labels
            y_test (np.ndarray): Testing labels
        
        Returns:
            acc (float): Classification accuracy on test set
            acc_shuff (float): Accuracy on shuffled test set, empirical chance
            conf_mat (np.ndarray, shape (n_labels,n_labels)): Confusion matrix for each condition
            confidence_scores (np.ndarray, shape (n_labels)): Confidence scores for each condition present in the test set
        """


        X_train, X_test = self._standardize(X_train, X_test)
        self.classifier.fit(X_train, y_train)

        acc = self.classifier.score(X_test, y_test) # get accuracy scores
        acc_shuff = self.classifier.score(X_test, self.rng.permutation(y_test)) # shuffled accuracy scores
        conf_mat = confusion_matrix(y_test, y_pred=self.classifier.predict(X_test),labels = self.labels) # actual and predicted confusion labels

        confidence_scores = np.full(self.n_labels, np.nan)
        confidence_scores_all = self.classifier.decision_function(X_test)
        for i, ss in enumerate(self.labels):
            confidence_scores[i] = confidence_scores_all[y_test == ss].mean() # get average score for each condition

        return acc, acc_shuff, conf_mat, confidence_scores

    def decode_across_time(self, X_train, X_test, y_train, y_test):
        """
        Main function to run decoding across each timepoint

        Args:
            X_train (np.ndarray, shape (n_bins,n_channels,n_times)): Training data
            X_test (np.ndarray, shape (n_bins,n_channels,n_times)): Testing data
            y_train (np.ndarray, shape (n_bins)): Training labels
            y_test (np.ndarray, shape (n_bins)): Testing labels


        Returns:
            accs (np.ndarray, shape (n_times)): Classification accuracy on test set for each timepoint
            accs_shuff (np.ndarray, shape (n_times)): Accuracy on shuffled test set, empirical chance for each timepoint
            conf_mats (np.ndarray, shape (n_labels,n_labels,n_times)): Confusion matrix for each condition for each timepoint
            confidence_scores (np.ndarray, shape (n_labels,n_times)): Confidence scores for each condition present in the test set for each timepoint
        

        """
        ntimes = X_train.shape[2]
        accs = np.full(ntimes, np.nan)
        accs_shuff = np.full(ntimes, np.nan)
        conf_mats = np.full((self.n_labels, self.n_labels, ntimes), np.nan)
        confidence_scores = np.full((self.n_labels, ntimes), np.nan)

        for itime in range(ntimes):
            (
                accs[itime],
                accs_shuff[itime],
                conf_mats[:, :, itime],
                confidence_scores[:, itime],
            ) = self._decode(X_train[:, :, itime], X_test[:, :, itime], y_train, y_test)

        return accs, accs_shuff, conf_mats, confidence_scores
