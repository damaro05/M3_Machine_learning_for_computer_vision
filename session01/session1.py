import numpy as np
import time
import data_loader
import evaluation
import os.path

from feature_extractor import FeatureExtractor
from classifier import Classifier

#######################
force_reload = False
feature_method = 'surf'
classifier = 'knn'
#######################

# Input images and labels
(train_images_filenames, train_labels) = data_loader.load_input_metadata('train_images_filenames.dat',
                                                                         'train_labels.dat')
(test_images_filenames, test_labels) = data_loader.load_input_metadata('test_images_filenames.dat', 'test_labels.dat')

print('Loaded ' + str(len(train_images_filenames)) + ' training images filenames with classes ', set(train_labels))
print('Loaded ' + str(len(test_images_filenames)) + ' testing images filenames with classes ', set(test_labels))

# Load precomputed labels if avaliable
precomp_label_filename = classifier + '_' + feature_method + '.npy'
if os.path.isfile(precomp_label_filename) and not force_reload:
    print 'Loading previous predictions'
    predicted_classes = np.load(precomp_label_filename)
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
        imfilename = test_images_filenames[i]
        des = fe.extract_single_image_features(imfilename)
        predictedclass = c.predict(des)
        predicted_classes.append(predictedclass)
        print('image ' + imfilename + ' was from class ' + test_labels[i] + ' and was predicted ' + predictedclass)
        numtestimages += 1

    end = time.time()
    print('Done in ' + str(end - start) + ' secs.')

    print 'Saving predicted clases'
    np.save(precomp_label_filename, predicted_classes)

numcorrect = np.sum(np.array(predicted_classes) == np.array(test_labels))
print('Final accuracy: ' + str(numcorrect * 100.0 / len(predicted_classes)))
precision, recall, fscore, support = evaluation.precision_recall_fscore(test_labels, predicted_classes)
average_precision, average_recall, average_fscore, average_support = evaluation.precision_recall_fscore(test_labels, predicted_classes, True)
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
print evaluation.confusionMatrix(test_labels, predicted_classes, True, True)

# classifier_probabilities = c.predict_proba(predicted_classes)
# evaluation.plot_roc_curve(test_labels, predicted_classes)
# evaluation.plot_roc_curve(test_labels, predicted_classes, c, classifier_probabilities)
# evaluation.plot_roc_curve(test_labels, predicted_classes, c)