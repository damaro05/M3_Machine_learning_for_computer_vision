import cv2
import numpy as np
import time
import data_loader
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

# Predict test set labels with classifier
numtestimages=0
numcorrect=0
for i in range(len(test_images_filenames)):
    filename=test_images_filenames[i]
    des = fe.extract_single_image_features(filename)
    predictedclass = c.predict(des)
    print('image ' + filename + ' was from class ' + test_labels[i] + ' and was predicted ' + predictedclass)
    numtestimages += 1
    if predictedclass == test_labels[i]:
        numcorrect += 1

# Performance
print('Final accuracy: ' + str(numcorrect*100.0/numtestimages))

end=time.time()
print('Done in '+str(end-start)+' secs.')