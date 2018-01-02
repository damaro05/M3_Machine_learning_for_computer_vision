import time
from sklearn.base import BaseEstimator
import numpy as np
from codebook import Codebook


class BoVWextractor(BaseEstimator):
    def __init__(self, K=512, no_dump_codebook=False):
        self.K = K
        self.no_dump_codebook=no_dump_codebook

    def fit(self, X, y=None):
        # Compute the codebook
        self.codebook = Codebook(K=self.K,no_dump=self.no_dump_codebook)
        self.codebook.fit(X)
        return self

    def transform(self, Train_descriptors):
        print 'Getting BoVW representation'
        init = time.time()

        visual_words = np.zeros((len(Train_descriptors), self.K), dtype=np.float32)
        for i in xrange(len(Train_descriptors)):
            words = self.codebook.predict(Train_descriptors[i])
            visual_words[i, :] = np.bincount(words, minlength=self.K)

        end = time.time()
        print '\tDone in ' + str(end - init) + ' secs.'
        return visual_words
