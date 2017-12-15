import numpy as np
import time
import data_loader
import evaluation
import os.path

from feature_extractor import FeatureExtractor
from classifier import Classifier

#######################
force_reload = False
feature_method = 'sift'
classifier = 'knn'
#######################

# Input images and labels
(train_images_filenames, train_labels) = data_loader.load_input_metadata('train_images_filenames.dat',
                                                                         'train_labels.dat')
(test_images_filenames, test_labels) = data_loader.load_input_metadata('test_images_filenames.dat', 'test_labels.dat')

print('Loaded ' + str(len(train_images_filenames)) + ' training images filenames with classes ', set(train_labels))
print('Loaded ' + str(len(test_images_filenames)) + ' testing images filenames with classes ', set(test_labels))

# Load precomputed labels if avaliable
filename = classifier + '_' + feature_method + '.npy'
if os.path.isfile(filename) and not force_reload:
    print 'Loading previous predictions'
    predicted_classes = np.load(filename)
else:
    start = time.time()

    print 'Extracting features'
    fe = FeatureExtractor(feature_method)
    (X, y) = fe.extract_features(train_images_filenames, train_labels, nimmax=30)

    print 'Training a classifier'
    c = Classifier(classifier)
    c.fit(X, y)

    print 'Predicting test set labels with the classifier'
    numtestimages = 0
    predicted_classes = []
    for i in range(len(test_images_filenames)):
        filename = test_images_filenames[i]
        des = fe.extract_single_image_features(filename)
        predictedclass = c.predict(des)
        predicted_classes.append(predictedclass)
        print('image ' + filename + ' was from class ' + test_labels[i] + ' and was predicted ' + predictedclass)
        numtestimages += 1

    end = time.time()
    print('Done in ' + str(end - start) + ' secs.')

    print 'Saving predicted clases'
    np.save(filename, predicted_classes)

numcorrect = np.sum(predicted_classes == test_labels)
print('Final accuracy: ' + str(numcorrect * 100.0 / len(predicted_classes)))
precision, recall, fscore, support = evaluation.precision_recall_fscore(test_labels, predicted_classes)
average_precision, _, _, _ = evaluation.precision_recall_fscore(test_labels, predicted_classes, True)
print 'Precision '
print precision
print 'Average precision'
print average_precision
print evaluation.confusionMatrix(test_labels, predicted_classes, True, True)

# classifier_probabilities = c.predict_proba(predicted_clases)
# evaluation.plot_roc_curve(test_labels, predicted_clases)
# evaluation.plot_roc_curve(test_labels, predicted_clases, c, classifier_probabilities)
# evaluation.plot_roc_curve(test_labels, predicted_clases, c)
