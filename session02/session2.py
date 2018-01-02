import time
from sklearn import metrics
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

# Definition of the pipeline
pipe = Pipeline([
    ('sift', SIFTextractor(nfeatures=300)),
    ('bovw', BoVWextractor(K=512)),
    ('scaler', StandardScaler()),
    ('svm', svm.SVC(kernel='rbf', C=1, gamma=.002)),
])

# Learning and classifying
pipe.fit(train_images_filenames, train_labels)
pred = pipe.predict(test_images_filenames)

print 'Final accuracy: ' + str(100 * metrics.accuracy_score(test_labels, pred)) + '%'

end = time.time()
print 'Everything done in ' + str(end - start) + ' secs.'
