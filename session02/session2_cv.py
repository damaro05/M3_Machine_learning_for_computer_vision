import gzip
import time

import cPickle

from datetime import datetime

import os
from sklearn.model_selection import GridSearchCV, KFold
from sklearn.pipeline import Pipeline
from sklearn import svm
from sklearn.preprocessing import StandardScaler

import data_loader
from bovw import BoVWextractor
from sift import SIFTextractor
from spatial_pyramids import SpatialPyramids
import svm_custom_kernels
from dense_sift import DenseSIFTextractor

start = time.time()

# Input images and labels
(train_images_filenames, train_labels) = data_loader.load_input_metadata('train_images_filenames.dat',
                                                                         'train_labels.dat')
(test_images_filenames, test_labels) = data_loader.load_input_metadata('test_images_filenames.dat', 'test_labels.dat')

print('Loaded ' + str(len(train_images_filenames)) + ' training images filenames with classes ', set(train_labels))
print('Loaded ' + str(len(test_images_filenames)) + ' testing images filenames with classes ', set(test_labels))

pipe = Pipeline([
    ('sift', DenseSIFTextractor()),
    ('spatial_pyramids', SpatialPyramids()),
    ('scaler', StandardScaler()),
    ('svm', svm.SVC(kernel=svm_custom_kernels.intersection_kernel)),
])

param_grid = {
    'sift__step': [10],
    'sift__radius': [5],
    'spatial_pyramids__K': [128, 256],
    'spatial_pyramids__num_levels': [2, 3],
    'svm__C': [4, 7],
    'svm__gamma': [.002]
}

grid = GridSearchCV(pipe, cv=3, n_jobs=1, param_grid=param_grid, scoring='accuracy',verbose=3)
grid.fit(train_images_filenames, train_labels)

if not os.path.exists('grids'):
    os.makedirs('grids')
with gzip.open('grids/grid_'+str(datetime.now()).replace(':','-')+'.pklz', 'wb') as f:
    print 'Dumping grid in ' + 'grids/grid_'+str(datetime.now())+'.pklz'
    cPickle.dump(grid, f, cPickle.HIGHEST_PROTOCOL)

for i in range(len(grid.cv_results_['mean_test_score'])):
    print str(grid.cv_results_['params'][i]) + ' -> ' + str(100 * grid.cv_results_['mean_test_score'][i]) + '%'

print 'Best score: ' + str(grid.best_score_)
print 'Best params: ' + str(grid.best_params_)

end = time.time()
print 'Everything done in ' + str(end - start) + ' secs.'
