import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import confusion_matrix


class Classifier:

    def __init__(self, labels, classifier=None, scaler=None):
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
        X_train = self.scaler.fit_transform(X_train)
        X_test = self.scaler.transform(X_test)
        return X_train, X_test

    def _get_acc(self, X_test, y_train, y_test):
        """
        Helper function that computes accuracy, but only for test trials whose labels were present in the training set
        """
        preds = self.classifier.predict(X_test)
        labeled_test = [i for i, y in enumerate(y_test) if y in np.unique(y_train)]
        acc = np.mean(preds[labeled_test] == y_test[labeled_test])
        return acc

    def _decode(self, X_train, X_test, y_train, y_test):
        X_train, X_test = self._standardize(X_train, X_test)
        self.classifier.fit(X_train, y_train)

        acc = self._get_acc(X_test, y_train, y_test)
        acc_shuff = self._get_acc(X_test, y_train, self.rng.permutation(y_test))
        conf_mat = confusion_matrix(y_test, y_pred=self.classifier.predict(X_test), labels=self.labels)

        confidence_scores = np.full(self.n_labels, np.nan)
        confidence_scores_all = self.classifier.decision_function(X_test)
        for i, ss in enumerate(self.labels):
            confidence_scores[i] = confidence_scores_all[y_test == ss].mean()

        return acc, acc_shuff, conf_mat, confidence_scores

    def decode_across_time(self, X_train, X_test, y_train, y_test):
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
