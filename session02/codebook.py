import os
import cPickle

import time
from sklearn.base import BaseEstimator
from sklearn.cluster import MiniBatchKMeans
import numpy as np


class Codebook(BaseEstimator):
    def __init__(self, K=512, picklepath='codebooks', force_reload=False, no_dump=False):
        self.picklepath = picklepath
        self.K = K
        self.force_reload = force_reload
        self.no_dump=no_dump

    def fit(self, descriptors_per_image, y=None):
        print 'Fitting the Codebook'
        init = time.time()

        # Obtain a hash representation of the input
        hashes = []
        for des in descriptors_per_image:
            des.flags.writeable = False
            hashes.append(hash(des.data))
        input_hash = hash(str(hashes))

        # Load precomputed data if avaliable
        self._dumpfile = 'codebook_' + str(self.K) + '_' + str(input_hash)
        if os.path.exists(self.picklepath + '/' + self._dumpfile + '.pklz') and not self.force_reload:
            with open(self.picklepath + '/' + self._dumpfile + '.pklz', 'rb') as f:
                print '\tLoading precomputed data in ' + self.picklepath + '/' + self._dumpfile + '_' + str(
                    input_hash) + '.pklz'
                self.clustering = cPickle.load(f)
                end = time.time()
                print '\tDone in ' + str(end - init) + ' secs.'
                return self

        # Initialize the clustering
        self.clustering = MiniBatchKMeans(n_clusters=self.K, verbose=False, batch_size=self.K * 20,
                                          compute_labels=False,
                                          reassignment_ratio=10 ** -4, random_state=42)

        # Put all the features in a single numpy array
        size_descriptors = descriptors_per_image[0].shape[1]
        all_descriptors = np.zeros((np.sum([len(p) for p in descriptors_per_image]), size_descriptors), dtype=np.uint8)
        startingpoint = 0
        for i in range(len(descriptors_per_image)):
            all_descriptors[startingpoint:startingpoint + len(descriptors_per_image[i])] = descriptors_per_image[i]
            startingpoint += len(descriptors_per_image[i])

        # Fit K-means
        self.clustering.fit(all_descriptors)

        # Dump data for a next execution
        if not self.no_dump:
            if not os.path.exists(self.picklepath):
                os.makedirs(self.picklepath)
            with open(self.picklepath + '/' + self._dumpfile + '.pklz', 'wb') as f:
                print '\tDumping data in ' + self.picklepath + '/' + self._dumpfile + '.pklz'
                cPickle.dump(self.clustering, f, cPickle.HIGHEST_PROTOCOL)

        end = time.time()
        print '\tDone in ' + str(end - init) + ' secs.'
        return self

    def predict(self, X):
        # K-means prediction
        return self.clustering.predict(X)
