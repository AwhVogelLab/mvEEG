import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import confusion_matrix


class Classifier:

    def __init__(self, classifier=None, scaler=None):
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
        scaler = StandardScaler()
        X_train = scaler.fit_transform(X_train)
        X_test = scaler.transform(X_test)
        return X_train, X_test

    def _decode(self, X_train, X_test, y_train, y_test):
        X_train, X_test = self._standardize(X_train, X_test)
        self.classifier.fit(X_train, y_train)

        acc = self.classifier.score(X_test, y_test)
        acc_shuff = self.classifier.score(X_test, self.rng.permutation(y_test))
        conf_mat = confusion_matrix(y_test, y_pred=self.classifier.predict(X_test))

        confidence_scores = np.full(len(set(y_test)), np.nan)
        confidence_scores_all = self.classifier.decision_function(X_test)
        for i, ss in enumerate(set(y_test)):
            confidence_scores[i] = confidence_scores_all[y_test == ss].mean()

        return acc, acc_shuff, conf_mat, confidence_scores

    def decode_across_time(self, X_train, X_test, y_train, y_test):
        ntimes = X_train.shape[2]
        cm_n = len(set(y_train) | set(y_test))
        accs = np.full(ntimes, np.nan)
        accs_shuff = np.full(ntimes, np.nan)
        conf_mats = np.full((cm_n, cm_n, ntimes), np.nan)
        confidence_scores = np.full((len(set(y_test)), ntimes), np.nan)

        for itime in range(ntimes):
            (
                accs[itime],
                accs_shuff[itime],
                conf_mats[:, :, itime],
                confidence_scores[:, itime],
            ) = self._decode(X_train[:, :, itime], X_test[:, :, itime], y_train, y_test)

        return accs, accs_shuff, conf_mats, confidence_scores
