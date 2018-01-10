import time
import numpy as np
from sklearn import metrics
from sklearn.pipeline import Pipeline
from sklearn import svm
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report

import data_loader
import evaluation
from bovw import BoVWextractor
from sift import SIFTextractor
from spatial_pyramids import SpatialPyramids
from dense_sift import DenseSIFTextractor

import svm_custom_kernels as ck

#######################
custom_kernel = True
confusion_matrix = True
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
    ('sift', DenseSIFTextractor(step=10, radius=5, max_nr_keypoints=1500)),
    ('spatial_pyramids', SpatialPyramids(K=256, num_levels=3)),
    ('scaler', StandardScaler()),
    ('svm', svm.SVC(kernel=kernel, C=4, gamma=.002)),
])

# Learning and classifying
pipe.fit(train_images_filenames, train_labels)
pred = pipe.predict(test_images_filenames)

# Report file
classes = ['mountain', 'inside_city', 'Opencountry', 'coast', 'street', 'forest', 'tallbuilding', 'highway']
file = open('report.txt', 'w')
file.write(classification_report(test_labels, pred, target_names=classes))
file.close()

print 'Final accuracy: ' + str(100 * metrics.accuracy_score(test_labels, pred)) + '%'

end = time.time()
print 'Everything done in ' + str(end - start) + ' secs.'

precision, recall, fscore, support = evaluation.precision_recall_fscore(test_labels, pred)
average_precision, average_recall, average_fscore, average_support = evaluation.precision_recall_fscore(test_labels, pred, True)
print 'Precision '
print np.round(precision*100,2)
print 'Average precision'
print round(average_precision*100,2)
print 'Recall '
print np.round(recall*100,2)
print 'Average recall'
print round(average_recall*100,2)
print 'F1'
print np.round(fscore*100,2)
print 'Average F1'
print round(average_fscore*100,2)

if confusion_matrix:
	print evaluation.confusionMatrix(test_labels, pred, True, True)