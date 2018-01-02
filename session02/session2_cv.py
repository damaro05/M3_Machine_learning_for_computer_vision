import gzip
import time

import cPickle

from datetime import datetime
from sklearn import metrics
from sklearn.model_selection import GridSearchCV, KFold
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn import svm
from sklearn.preprocessing import StandardScaler

import data_loader
from bovw import BoVWextractor
from sift import SIFTextractor

start = time.time()

# Input images and labels
(train_images_filenames, train_labels) = data_loader.load_input_metadata('train_images_filenames.dat',
                                                                         'train_labels.dat')
(test_images_filenames, test_labels) = data_loader.load_input_metadata('test_images_filenames.dat', 'test_labels.dat')

print('Loaded ' + str(len(train_images_filenames)) + ' training images filenames with classes ', set(train_labels))
print('Loaded ' + str(len(test_images_filenames)) + ' testing images filenames with classes ', set(test_labels))

pipe = Pipeline([
    ('sift', SIFTextractor()),
    ('bovw', BoVWextractor()),
    ('scaler', StandardScaler()),
    ('svm', svm.SVC(kernel='rbf')),
])

param_grid = {
    'sift__nfeatures': [300],
    'bovw__K': [512, 1024],
    'svm__C': [1, 2],
    'svm__gamma': [.002]
}

grid = GridSearchCV(pipe, cv=3, n_jobs=1, param_grid=param_grid, scoring='accuracy') #cv=KFold(n_splits=3,random_state=0)
grid.fit(train_images_filenames, train_labels)

with gzip.open('grid_'+str(datetime.now())+'.pklz', 'wb') as f:
    print 'Dumping grid in ' + 'grid'+str(time.clock())+'.pklz'
    cPickle.dump(grid, f, cPickle.HIGHEST_PROTOCOL)

for i in range(len(grid.cv_results_['mean_test_score'])):
    print str(grid.cv_results_['params'][i]) + ' -> ' + str(grid.cv_results_['mean_test_score'][i])

print 'Best score: ' + str(grid.best_score_)
print 'Best params: ' + str(grid.best_params_)

#pipe.fit(train_images_filenames, train_labels)
#pred = pipe.predict(test_images_filenames)

#print 'Final accuracy: ' + str(100 * metrics.accuracy_score(test_labels, pred))

end = time.time()
print 'Everything done in ' + str(end - start) + ' secs.'
