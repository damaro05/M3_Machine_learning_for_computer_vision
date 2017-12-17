from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn import tree
from sklearn import svm
from logistic_regression import LogisticRegression
import sys
import numpy as np


class Classifier:
    def __init__(self, method):
        if method == 'knn':
            self.name = 'knn_classifier'
            self.fit = self._knn_fit
            self.predict = self._knn_predict
            self.predict_proba = self._knn_predict_proba
        elif method == 'random_forest':
            self.name = 'random_forest_classifier'
            self.fit = self._randomf_fit
            self.predict = self._randomf_predict
            self.predict_proba = self._randomf_predict_proba
        elif method == 'bayes':
            self.name = 'naive_bayes_classifier'
            self.fit = self._bayes_fit
            self.predict = self._bayes_predict
            self.predict_proba = self._bayes_predict_proba
        elif method == 'tree':
            self.name = 'decision_tree_classifier'
            self.fit = self._tree_fit
            self.predict = self._tree_predict
            self.predict_proba = self._tree_predict_proba
        elif method == 'svc':
            self.name = 'support_vector_classification'
            self.fit = self._svc_fit
            self.predict = self._svc_predict
            self.predict_proba = self._svc_predict_proba
        elif method == 'linearsvc':
            self.name = 'linear_support_vector_classification'
            self.fit = self._lsvc_fit
            self.predict = self._lsvc_predict
            self.predict_proba = self._lsvc_predict_proba
        elif method == 'logisticregression':
            self.name = 'logistic_regression'
            self.fit = self._lr_fit
            self.predict = self._lr_predict
            #self.predict_proba = self._lsvc_predict_proba
        else:
            print('Classifying method not found')
            sys.exit(-1)

    def _knn_fit(self, X, y):
        print('Training the knn classifier...')
        self._classifier = KNeighborsClassifier(n_neighbors=5, n_jobs=-1)
        self._classifier.fit(X, y)
        print('Done!')

    def _knn_predict(self, X):
        predictions = self._classifier.predict(X)
        values, counts = np.unique(predictions, return_counts=True)
        return values[np.argmax(counts)]

    def _knn_predict_proba(self, X):
        pred_probabilities = self._classifier.predict_proba(X)
        print 'Predicted probabilities'
        return pred_probabilities

    def _lr_fit(self, X, y):
        print('Training the logistic regression...')
        self._classifier = LogisticRegression()
        self._classifier.fit(X, y)
        print('Done!')

    def _lr_predict(self, X):
        predictions = self._classifier.predict(X)
        values, counts = np.unique(predictions, return_counts=True)
        return values[np.argmax(counts)]

    def _randomf_fit(self, X, y):
        print('Training the Random forest classifier...')
        self._classifier = RandomForestClassifier(max_depth=3, random_state=0, n_jobs=-1)
        self._classifier.fit(X, y)
        print('Done!')

    def _randomf_predict(self, X):
        predictions = self._classifier.predict(X)
        values, counts = np.unique(predictions, return_counts=True)
        return values[np.argmax(counts)]

    def _randomf_predict_proba(self, X):
        pred_probabilities = self._classifier.predict_proba(X)
        print 'Predicted probabilities'
        return pred_probabilities

    def _bayes_fit(self, X, y):
        print('Training the Gaussian Naive Bayes classifier...')
        self._classifier = GaussianNB()
        self._classifier.fit(X, y)
        print('Done!')

    def _bayes_predict(self, X):
        predictions = self._classifier.predict(X)
        values, counts = np.unique(predictions, return_counts=True)
        return values[np.argmax(counts)]

    def _bayes_predict_proba(self, X):
        pred_probabilities = self._classifier.predict_proba(X)
        print 'Predicted probabilities'
        print pred_probabilities
        return pred_probabilities

    def _tree_fit(self, X, y):
        print('Training the Decision tree classifier...')
        self._classifier = tree.DecisionTreeClassifier()
        self._classifier.fit(X, y)
        print('Done!')

    def _tree_predict(self, X):
        predictions = self._classifier.predict(X)
        values, counts = np.unique(predictions, return_counts=True)
        return values[np.argmax(counts)]

    def _tree_predict_proba(self, X):
        pred_probabilities = self._classifier.predict_proba(X)
        print 'Predicted probabilities'
        return pred_probabilities

    def _svc_fit(self, X, y):
        print('Training the Support Vector classifier...')
        # ovr = one vs. rest | ovo = one vs. one
        self._classifier = svm.SVC(decision_function_shape='ovr')
        self._classifier.fit(X, y)
        print('Done!')

    def _svc_predict(self, X):
        predictions = self._classifier.predict(X)
        values, counts = np.unique(predictions, return_counts=True)
        return values[np.argmax(counts)]

    def _svc_predict_proba(self, X):
        pred_probabilities = self._classifier.predict_proba(X)
        print 'Predicted probabilities'
        return pred_probabilities

    def _lsvc_fit(self, X, y):
        print('Training the Linear Support Vector classifier...')
        self._classifier = svm.LinearSVC()
        self._classifier.fit(X, y)
        print('Done!')

    def _lsvc_predict(self, X):
        predictions = self._classifier.predict(X)
        values, counts = np.unique(predictions, return_counts=True)
        return values[np.argmax(counts)]

    def _lsvc_predict_proba(self, X):
        pred_probabilities = self._classifier.predict_proba(X)
        print 'Predicted probabilities'
        return pred_probabilities