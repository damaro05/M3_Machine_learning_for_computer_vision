import time
from sklearn.base import BaseEstimator
import numpy as np
from codebook import Codebook


class BoVWextractor(BaseEstimator):
    def __init__(self, K=512, no_dump_codebook=False, force_reload=False):
        self.K = K
        self.no_dump_codebook=no_dump_codebook
        self.force_reload = force_reload

    def fit(self, X, y=None):
        # Compute the codebook
        self.codebook = Codebook(K=self.K,no_dump=self.no_dump_codebook, force_reload=self.force_reload)
        self.codebook.fit(X['descriptors'])
        return self

    def transform(self, X):
        print 'Getting BoVW representation'
        init = time.time()

        descriptors = X['descriptors']

        visual_words = np.zeros((len(descriptors), self.K), dtype=np.float32)
        for i in xrange(len(descriptors)):
            words = self.codebook.predict(descriptors[i])
            visual_words[i, :] = np.bincount(words, minlength=self.K)

        end = time.time()
        print '\tDone in ' + str(end - init) + ' secs.'
        return visual_words
