import time
from sklearn.base import BaseEstimator
import numpy as np
from codebook import Codebook


class SpatialPyramids(BaseEstimator):
    def __init__(self, K=512, num_levels=3, no_dump_codebook=False):
        self.K = K
        self.num_levels = num_levels
        self.no_dump_codebook = no_dump_codebook

    def fit(self, X, y=None):
        # Compute the codebook
        self.codebook = Codebook(K=self.K, no_dump=self.no_dump_codebook)
        self.codebook.fit(X['descriptors'])
        return self

    def transform(self, X):
        print 'Getting Spatial Pyramid representation'
        init = time.time()

        descriptors = X['descriptors']
        positions = X['positions']
        imsizes = X['imsizes']

        # Num. of cols/rows for each pyramid level
        grid_ncolsrows = 2 ** np.arange(self.num_levels)

        visual_words = np.zeros((len(descriptors), self.K * np.sum(grid_ncolsrows ** 2)), dtype=np.float32)
        for im in xrange(len(descriptors)):
            # Compute the words
            words = self.codebook.predict(descriptors[im])

            # Compute the bincount for each grid cell in each pyramid level
            current_vw = []
            for l in range(self.num_levels):
                r_vec = np.linspace(0, imsizes[im][0] + 1, num=grid_ncolsrows[l] + 1)
                c_vec = np.linspace(0, imsizes[im][1] + 1, num=grid_ncolsrows[l] + 1)
                for i in range(grid_ncolsrows[l]):
                    for j in range(grid_ncolsrows[l]):
                        rb = np.logical_and(positions[im][:, 0] >= r_vec[i], positions[im][:, 0] < r_vec[i + 1])
                        cb = np.logical_and(positions[im][:, 1] >= c_vec[j], positions[im][:, 1] < c_vec[j + 1])
                        current_vw.extend(np.bincount(words[np.logical_and(rb, cb)], minlength=self.K))

            # Save the computed values
            visual_words[im, :] = current_vw

        end = time.time()
        print '\tDone in ' + str(end - init) + ' secs.'
        return visual_words
