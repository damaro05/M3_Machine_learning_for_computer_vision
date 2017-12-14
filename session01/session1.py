import cv2
import numpy as np
import time
import data_loader
import evaluation
import os.path

from feature_extractor import FeatureExtractor
from classifier import Classifier

start = time.time()

# Input images and labels
(train_images_filenames,train_labels)=data_loader.load_input_metadata('train_images_filenames.dat', 'train_labels.dat')
(test_images_filenames,test_labels)=data_loader.load_input_metadata('test_images_filenames.dat', 'test_labels.dat')

print('Loaded '+str(len(train_images_filenames))+' training images filenames with classes ',set(train_labels))
print('Loaded '+str(len(test_images_filenames))+' testing images filenames with classes ',set(test_labels))

# Feature extraction
fe=FeatureExtractor('sift')
(D,L)=fe.extract_features(train_images_filenames,train_labels,nimmax=30)

# Train a classifier
c=Classifier('knn')
c.fit(D,L)
filename = c.name + '.npy'
#c.__dict__.keys()

# Predict test set labels with classifier
numtestimages=0
numcorrect=0
predicted_clases = []

if not os.path.isfile(filename):
	for i in range(len(test_images_filenames)):
	    filename=test_images_filenames[i]
	    des = fe.extract_single_image_features(filename)
	    predictedclass = c.predict(des)
	    predicted_clases.append(predictedclass)
	    #print('image ' + filename + ' was from class ' + test_labels[i] + ' and was predicted ' + predictedclass)
	    numtestimages += 1
	    if predictedclass == test_labels[i]:
	        numcorrect += 1
	
	print 'Saving predicted clases'
	np.save(filename, predicted_clases)
	print('Final accuracy: ' + str(numcorrect*100.0/numtestimages))
else:
	print 'Loading previous predictions'
	predicted_clases = np.load(filename)

# Performance
# guardar numcorrect and next es equivalente
#print('Final accuracy: ' + str(numcorrect*100.0/len(predicted_clases)))

print evaluation.confusionMatrix(test_labels, predicted_clases, True, True)
precision, recall, fscore, support = evaluation.precision_recall_fscore(test_labels, predicted_clases)
average_precision, _, _, _ = evaluation.precision_recall_fscore(test_labels, predicted_clases, True)
print 'Precision '
print precision
print 'Average precision'
print average_precision

#classifier_probabilities = c.predict_proba(predicted_clases)
#evaluation.plot_roc_curve(test_labels, predicted_clases)
#evaluation.plot_roc_curve(test_labels, predicted_clases, c, classifier_probabilities)
#evaluation.plot_roc_curve(test_labels, predicted_clases, c)
end=time.time()
print('Done in '+str(end-start)+' secs.')