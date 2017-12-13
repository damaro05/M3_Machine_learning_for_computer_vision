import numpy as np
import cv2
import sys
import os
import pickle #module for serialization of python object structure
import gzip   #we can compress our data directly when writting data to a file

class FeatureExtractor:
    def __init__(self, method):
        self.method=method
        if method == 'sift':
            self.f = self._sift_features
        else:
            print('Feature extracting method not found')
            sys.exit(-1)

    def _sift_features(self, ima):
        # SIFTdetector = cv2.SIFT(nfeatures=100)
        SIFTdetector = cv2.xfeatures2d.SIFT_create(nfeatures=100)
        gray = cv2.cvtColor(ima, cv2.COLOR_BGR2GRAY)
        kpt, des = SIFTdetector.detectAndCompute(gray, None)
        print(str(len(kpt)) + ' extracted keypoints and descriptors')
        return des

    def extract_features(self, train_images_filenames, train_labels, nimmax=float('inf'),picklepath='../../FeaturePickles'):
        # If features were previously computed, load them directly
        if os.path.exists(picklepath+'/'+self.method+'.pklz'):
            with gzip.open(picklepath+'/'+self.method+'.pklz', 'rb') as f:
                (D, L) = pickle.load(f)
                return (D, L)

        # If not ...
        Train_descriptors = []
        Train_label_per_descriptor = []

        for i in range(len(train_images_filenames)):
            filename = train_images_filenames[i]
            if Train_label_per_descriptor.count(train_labels[i]) < nimmax:
                print('Reading image ' + filename)
                ima = cv2.imread(filename)
                des = self.f(ima)
                Train_descriptors.append(des)
                Train_label_per_descriptor.append(train_labels[i])

        # Transform everything to numpy arrays
        D = Train_descriptors[0]
        L = np.array([Train_label_per_descriptor[0]] * Train_descriptors[0].shape[0])

        for i in range(1, len(Train_descriptors)):
            D = np.vstack((D, Train_descriptors[i]))
            L = np.hstack((L, np.array([Train_label_per_descriptor[i]] * Train_descriptors[i].shape[0])))

        # Save features in a pickle
        if not os.path.exists(picklepath):
            os.makedirs(picklepath)
        with gzip.open(picklepath+'/'+self.method+'.pklz','wb') as f:
            pickle.dump((D,L),f,pickle.HIGHEST_PROTOCOL)

        return (D, L)

    def extract_single_image_features(self, image_filename):
        ima = cv2.imread(image_filename)
        des = self.f(ima)
        return des
