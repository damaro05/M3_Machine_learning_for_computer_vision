import numpy as np
import cv2
import sys
import os
import pickle #module for serialization of python object structure
import gzip   #we can compress our data directly when writting data to a file
from skimage.feature import hog
from sklearn.cluster import KMeans
from skimage.exposure import histogram
import matplotlib.pyplot as plt


class FeatureExtractor:
    def __init__(self, method):
        self.method=method
        if method == 'sift':
            self.f = self._sift_features
        if method == 'aggsift':
            self.f = self._aggregated_sift_features
        elif method == 'surf':
            self.f = self._surf_features
        elif method == 'hog':
            self.f = self._hog_features
        elif method == 'kmeans':
            self.f = self._kmeans_features
        elif method == 'histogram':
            self.f = self._histogram_features
        elif method == 'aggsift_histogram':
            self.f = self._aggsift_histogram_features
        else:
            print('Feature extracting method not found')
            sys.exit(-1)

    def _aggsift_histogram_features(self, ima):
        SIFTdetector = cv2.xfeatures2d.SIFT_create(nfeatures=100)
        gray = cv2.cvtColor(ima, cv2.COLOR_BGR2GRAY)
        _, des = SIFTdetector.detectAndCompute(gray, None)
        #print(str(len(kpt)) + ' extracted keypoints and descriptors')
        sift = np.mean(des,axis=0)

        hsv = cv2.cvtColor(ima, cv2.COLOR_BGR2HSV)
        pixels = np.reshape(hsv, (ima.shape[0] * ima.shape[1], 1, 3))
        pixels = np.squeeze(np.transpose(pixels, (0, 2, 1)))
        # Color
        hpixels = pixels[np.logical_and(pixels[:,1] >= 25,pixels[:,2] >= 25),0]
        hhist,_=np.histogram(hpixels, bins=12)
        gpixels = pixels[pixels[:, 1] < 25, 2]
        ghist, _ = np.histogram(gpixels, bins=5)

        return np.concatenate([hhist,ghist,sift])[np.newaxis]

    def _kmeans_features(self, ima):
        kmeans = KMeans(n_clusters=4)
        hsv = cv2.cvtColor(ima, cv2.COLOR_BGR2HSV)
        pixels = np.reshape(hsv, (ima.shape[0]*ima.shape[1], 1, 3))
        pixels = np.squeeze(np.transpose(pixels, (0, 2, 1)))
        kmeans.fit(pixels)
        return np.reshape(kmeans.cluster_centers_,(1,4*3))

    def _histogram_features(self, ima):
        hsv = cv2.cvtColor(ima, cv2.COLOR_BGR2HSV)
        pixels = np.reshape(hsv, (ima.shape[0] * ima.shape[1], 1, 3))
        pixels = np.squeeze(np.transpose(pixels, (0, 2, 1)))
        # Color
        hpixels = pixels[np.logical_and(pixels[:,1] >= 25,pixels[:,2] >= 25),0]
        hhist,_=np.histogram(hpixels, bins=12)
        gpixels = pixels[pixels[:, 1] < 25, 2]
        ghist, _ = np.histogram(gpixels, bins=5)

        return np.concatenate([hhist,ghist])[np.newaxis]

    def _sift_features(self, ima):
        # SIFTdetector = cv2.SIFT(nfeatures=100)
        SIFTdetector = cv2.xfeatures2d.SIFT_create(nfeatures=100)
        gray = cv2.cvtColor(ima, cv2.COLOR_BGR2GRAY)
        kpt, des = SIFTdetector.detectAndCompute(gray, None)
        print(str(len(kpt)) + ' extracted keypoints and descriptors')
        return des

    def _aggregated_sift_features(self, ima):
        # SIFTdetector = cv2.SIFT(nfeatures=100)
        SIFTdetector = cv2.xfeatures2d.SIFT_create(nfeatures=100)
        gray = cv2.cvtColor(ima, cv2.COLOR_BGR2GRAY)
        kpt, des = SIFTdetector.detectAndCompute(gray, None)
        print(str(len(kpt)) + ' extracted keypoints and descriptors')
        return np.mean(des,axis=0)[np.newaxis]

    def _surf_features(self, ima):
        SURFdetector = cv2.xfeatures2d.SURF_create(400)
        gray = cv2.cvtColor(ima, cv2.COLOR_BGR2GRAY)
        kpt, des = SURFdetector.detectAndCompute(gray, None)
        print(str(len(kpt)) + ' extracted keypoints and descriptors')
        return des

    def _hog_features(self, ima):
        #hog = cv2.HOGDescriptor()
        #h = hog.compute(ima)
        gray = cv2.cvtColor(ima, cv2.COLOR_BGR2GRAY)
        #from skimage import exposure
        #import matplotlib.pyplot as plt
        #fd, hog_image = hog(gray, visualise=True,pixels_per_cell=(16,16))
        #hog_image_rescaled = exposure.rescale_intensity(hog_image, in_range=(0, 10))
        #plt.imshow(hog_image_rescaled, cmap=plt.cm.gray)

        fd = hog(gray,pixels_per_cell=(16,16)) # Adjust pixels_per_cell by a K-fold
        return fd[np.newaxis]

    def extract_features(self, train_images_filenames, train_labels, nimmax=float('inf'),picklepath='../../FeaturePickles', force_reload=False):
        # If features were previously computed, load them directly
        if os.path.exists(picklepath+'/'+self.method+'.pklz') and not force_reload:
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
