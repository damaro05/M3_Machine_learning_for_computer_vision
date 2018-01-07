import time
from sklearn.base import BaseEstimator
import numpy as np
from codebook import Codebook
from sklearn.decomposition import PCA
from yael import ynumpy

class FisherVectors(BaseEstimator):
    def __init__(self, K=512, n_components=64, no_dump_codebook=False):
        self.K = K
        self.num_levels = num_levels
        self.no_dump_codebook = no_dump_codebook
        self.n_components=n_components

    def fit(self, X, y=None):
        image_descs = X['descriptors']

        # Compute a PCA over the SIFT descriptors
        # make a big matrix with all image descriptors
        all_desc = np.vstack(image_descs)
        '''
        #n_sample = k * 1000

        # choose n_sample descriptors at random
        sample_indices = np.random.choice(all_desc.shape[0], n_sample)
        sample = all_desc[sample_indices]

        # until now sample was in uint8. Convert to float32
        sample = sample.astype('float32')

        # compute mean and covariance matrix for the PCA
        mean = sample.mean(axis=0)
        sample = sample - mean
        cov = np.dot(sample.T, sample)

        # compute PCA matrix and keep only 64 dimensions
        eigvals, eigvecs = np.linalg.eig(cov)
        perm = eigvals.argsort()  # sort by increasing eigenvalue
        pca_transform = eigvecs[:, perm[64:128]]  # eigenvectors for the 64 last eigenvalues
        
        # transform sample with PCA (note that numpy imposes line-vectors,
        # so we right-multiply the vectors)
        sample = np.dot(sample, pca_transform)
        '''
        self.mean = np.mean(all_desc, axis=0)
        all_desc=all_desc-self.mean

        self.PCA = PCA(n_components=self.n_components)
        self.PCA.fit(all_desc)
        all_desc=self.PCA.transform(all_desc)

        # train GMM Codebook
        self.gmm = ynumpy.gmm_learn(all_desc, self.K)

        return self

    def transform(self, X):
        print 'Getting Fisher Vector representation'
        init = time.time()

        descriptors = X['descriptors']
        positions = X['positions']
        imsizes = X['imsizes']
        image_fvs=[]
        for image_desc in descriptors:
            # apply the PCA to the image descriptor
            image_desc = self.PCA.transform(image_desc-self.mean)
            # compute the Fisher vector, using only the derivative w.r.t mu
            fv = ynumpy.fisher(self.gmm, image_desc, include='mu')
            image_fvs.append(fv)

        end = time.time()
        print '\tDone in ' + str(end - init) + ' secs.'
        return image_fvs
