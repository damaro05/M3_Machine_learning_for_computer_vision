import time

import cPickle
from sklearn.base import BaseEstimator
import cv2
import os
# import gzip
import numpy as np
import pickle


class SIFTextractor(BaseEstimator):
    def __init__(self, nfeatures=300, picklepath='../../FeaturePickles', force_reload=False, no_dump=False, dense=False, options={}):
        # Dense sift: SIFTdetector.compute(image,keypoints)
        self.nfeatures = nfeatures
        self.picklepath = picklepath
        self.force_reload = force_reload
        self._dumpfile = 'SIFT_features_' + str(nfeatures)
        self.no_dump = no_dump
        self.dense = dense
        self.options = options

    def fit(self, X, y=None):
        return self

    def transform(self, images_filenames):
        print 'Getting the SIFT features'
        init = time.time()

        # Load precomputed data if avaliable
        if os.path.exists(self.picklepath + '/' + self._dumpfile + '_' + str(
                hash(str(images_filenames))) + '.pklz') and not self.force_reload:
            with open(self.picklepath + '/' + self._dumpfile + '_' + str(hash(str(images_filenames))) + '.pklz',
                      'rb') as f:
                print '\tLoading precomputed data in ' + self.picklepath + '/' + self._dumpfile + '_' + str(
                    hash(str(images_filenames))) + '.pklz'
                descriptors_per_image = cPickle.load(f)
                end = time.time()
                print '\tDone in ' + str(end - init) + ' secs.'
                return descriptors_per_image

        # Extract SIFT keypoints and descriptors for each image
        SIFTdetector = cv2.xfeatures2d.SIFT_create(nfeatures=self.nfeatures)
        descriptors_per_image = []
        kptpositions_per_image = []
        image_sizes = []
        for i in range(len(images_filenames)):
            filename = images_filenames[i]
            # print 'Reading image ' + filename
            ima = cv2.imread(filename)
            gray = cv2.cvtColor(ima, cv2.COLOR_BGR2GRAY)
            # Check Dense sampling
            if self.dense:
                kpt = self.dense_sampling(self.options['max_nr_keyp'], self.options['keyp_step'], self.options['keyp_radius'], \
                                     gray.shape[0], gray.shape[1])
                kpt, des = SIFTdetector.compute(gray, kpt)
            else:
                kpt, des = SIFTdetector.detectAndCompute(gray, None)
            pos = [auxkpt.pt for auxkpt in kpt]
            descriptors_per_image.append(des)
            kptpositions_per_image.append(np.array(pos))
            image_sizes.append(ima.shape)
            # print str(len(kpt)) + ' extracted keypoints and descriptors'

        data = {'descriptors': descriptors_per_image, 'positions': kptpositions_per_image, 'imsizes': image_sizes}

        # Dump data for a next execution
        if not self.no_dump:
            if not os.path.exists(self.picklepath):
                os.makedirs(self.picklepath)
            with open(self.picklepath + '/' + self._dumpfile + '_' + str(hash(str(images_filenames))) + '.pklz',
                      'wb') as f:
                print '\tDumping data in ' + self.picklepath + '/' + self._dumpfile + '_' + str(
                    hash(str(images_filenames))) + '.pklz'
                cPickle.dump(data, f, pickle.HIGHEST_PROTOCOL)

        end = time.time()
        print '\tDone in ' + str(end - init) + ' secs.'
        return data

    def dense_sampling(self, max_nr_keypoints, step_size, radius, image_height, image_width):
        nr_keypoints = (image_height/step_size)*(image_width/step_size)
        while not nr_keypoints <= max_nr_keypoints:
            step_size = step_size - 1
            if step_size < 1:
                step_size = 1
                nr_keypoints = (image_height / step_size) * (image_width / step_size)
                break

        if step_size < 1:
            step_size = 1

        kpt = [cv2.KeyPoint(x, y, radius) for y in range(step_size-1, image_height-step_size, step_size)
                                        for x in range(step_size-1, image_width-step_size, step_size)]

        return kpt
