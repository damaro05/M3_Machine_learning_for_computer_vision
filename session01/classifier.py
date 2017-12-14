from sklearn.neighbors import KNeighborsClassifier
import sys
import numpy as np


class Classifier:
    def __init__(self, method):
        if method == 'knn':
            self.name = 'knn_classifier'
            self.fit = self._knn_fit
            self.predict = self._knn_predict
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
