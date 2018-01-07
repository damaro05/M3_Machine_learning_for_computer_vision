import time
from sklearn import metrics
from sklearn.pipeline import Pipeline
from sklearn import svm
from sklearn.preprocessing import StandardScaler

import data_loader
from bovw import BoVWextractor
from sift import SIFTextractor
from spatial_pyramids import SpatialPyramids

import svm_custom_kernels as ck

#######################
custom_kernel = False

# SIFT options
sift_dense = True
sift_options = {}
sift_options['max_nr_keyp'] = 1500	# Max number of equally spaced keypoints
sift_options['keyp_step'] = 10		# Step size
sift_options['keyp_radius'] = 5		# Radius size
#######################

start = time.time()

# Input images and labels
(train_images_filenames, train_labels) = data_loader.load_input_metadata('train_images_filenames.dat',
                                                                         'train_labels.dat')
(test_images_filenames, test_labels) = data_loader.load_input_metadata('test_images_filenames.dat', 'test_labels.dat')

print('Loaded ' + str(len(train_images_filenames)) + ' training images filenames with classes ', set(train_labels))
print('Loaded ' + str(len(test_images_filenames)) + ' testing images filenames with classes ', set(test_labels))

kernel = 'rbf'
if custom_kernel:
	kernel = ck.intersection_kernel

# Definition of the pipeline
pipe = Pipeline([
    ('sift', SIFTextractor(nfeatures=300, dense=sift_dense, options=sift_options)),
    ('bovw', BoVWextractor(K=512)),
    ('scaler', StandardScaler()),
    ('svm', svm.SVC(kernel=kernel, C=1, gamma=.002)),
])

# Learning and classifying
pipe.fit(train_images_filenames, train_labels)
pred = pipe.predict(test_images_filenames)

print 'Final accuracy: ' + str(100 * metrics.accuracy_score(test_labels, pred)) + '%'

end = time.time()
print 'Everything done in ' + str(end - start) + ' secs.'
