import matplotlib
from PIL import Image
from keras import Model
from sklearn import svm

from patches.bovw import BoVWextractor

from patches.patch_mlp_builder import build_patch_mlp_svm

matplotlib.use('Agg')
import time

import cPickle

import numpy as np
from skimage.io import imread
from sklearn.feature_extraction.image import extract_patches_2d
from sklearn.metrics import accuracy_score

from patches.patch_data_loader import build_patch_generator
from utils import colorprint, Color, softmax, generate_image_patches_db, plot_history
import os

## PARAMETERS ##########################################################################################################
PATCH_SIZE = 48
EPOCHS = 50
MAX_PATCHES = 32
BATCH_SIZE = MAX_PATCHES * 32
LAYER = 'second'
K = 512
DATASET_DIR = '/share/datasets/MIT_split'
PATCHES_DIR = '/home/master04/data/MIT_split_patches_' + str(PATCH_SIZE)
CLASSES = ['coast', 'forest', 'highway', 'inside_city', 'mountain', 'Opencountry', 'street', 'tallbuilding']
RECOMPUTE = False
TRAIN_WITH_VALIDATION = True
########################################################################################################################

# Check the location of the dataset
if not os.path.exists(DATASET_DIR):
    colorprint(Color.RED, 'ERROR: dataset directory ' + DATASET_DIR + ' do not exists!\n')
    quit()

# Check the existence of the patches dataset
if not os.path.exists(PATCHES_DIR):
    colorprint(Color.YELLOW, 'WARNING: patches dataset directory ' + PATCHES_DIR + ' do not exists!\n')
    colorprint(Color.BLUE, 'Creating image patches dataset into ' + PATCHES_DIR + '\n')
    generate_image_patches_db(DATASET_DIR, PATCHES_DIR, patch_size=PATCH_SIZE, max_patches=MAX_PATCHES)
    colorprint(Color.BLUE, 'Done!\n')

# Build the NN
model = build_patch_mlp_svm(PATCH_SIZE)

# Train or load the weights
model_identifier = str(hash(str(model.get_config()))) + '_' + str(PATCH_SIZE) + '_' + str(BATCH_SIZE) + '_' + str(
    EPOCHS) + '_' + str(MAX_PATCHES)
if os.path.exists('dump/patch_models/' + model_identifier + '.h5') and not RECOMPUTE:
    model.load_weights('dump/patch_models/' + model_identifier + '.h5')
    colorprint(Color.GREEN, 'Loading model file dump/patch_models/' + model_identifier + '.h5\n')
else:
    if not os.path.exists('dump/patch_models'):
        os.mkdir('dump/patch_models')
    if os.path.exists(model_identifier + '.h5'):
        colorprint(Color.YELLOW,
                   'WARNING: model file dump/patch_models/' + model_identifier + '.h5 exists and will be overwritten!\n')

    # this is a generator that will read pictures found in
    # subfolers of 'data/train', and indefinitely generate
    # batches of augmented image data
    train_generator = build_patch_generator(PATCHES_DIR, PATCH_SIZE, BATCH_SIZE, SPLIT='train', CLASSES=CLASSES)

    # this is a similar generator, for validation data
    if TRAIN_WITH_VALIDATION:
        validation_generator = build_patch_generator(PATCHES_DIR, PATCH_SIZE, BATCH_SIZE, SPLIT='test', CLASSES=CLASSES)
        validation_steps = validation_generator.samples // BATCH_SIZE
    else:
        validation_generator = None
        validation_steps = None

    colorprint(Color.BLUE, 'Start training...\n')
    start = time.clock()

    history = model.fit_generator(
        train_generator,
        steps_per_epoch=train_generator.samples // BATCH_SIZE,
        epochs=EPOCHS,
        validation_data=validation_generator,
        validation_steps=validation_steps)

    end = time.clock()
    colorprint(Color.BLUE, 'Done! Elapsed time: ' + str(end - start) + 'sec\n')
    colorprint(Color.BLUE, 'Saving the model into ' + model_identifier + '.h5 \n')
    model.save_weights(
        'dump/patch_models/' + model_identifier + '.h5')  # always save your weights after training or during training
    colorprint(Color.BLUE, 'Done!\n')
    if not os.path.exists('dump/patch_histories'):
        os.mkdir('dump/patch_histories')
    with open('dump/patch_histories/' + model_identifier + '_history.pklz', 'wb') as f:
        cPickle.dump(
            (history.epoch, history.history, history.params, history.validation_data, model.get_config()), f,
            cPickle.HIGHEST_PROTOCOL)

    # summarize history for accuracy
    plot_history(history, model_identifier, metric='acc', plot_validation=TRAIN_WITH_VALIDATION, path='dump/patch_histories/')
    # summarize history for loss
    plot_history(history, model_identifier, metric='loss', plot_validation=TRAIN_WITH_VALIDATION, path='dump/patch_histories/')

colorprint(Color.BLUE, 'Start generating BoVW representation...\n')
start = time.clock()

model_layer = Model(inputs=model.input, outputs=model.get_layer(LAYER).output)

CLASSES = np.array(CLASSES)

y_train = []
X_train = []
for cls in CLASSES:
    for imname in os.listdir(os.path.join(DATASET_DIR, 'train', cls)):
        im = Image.open(os.path.join(DATASET_DIR, 'train', cls, imname))
        y_train.append(cls)
        patches = extract_patches_2d(np.array(im), (PATCH_SIZE, PATCH_SIZE), max_patches=MAX_PATCHES)
        out = model_layer.predict(patches)
        X_train.append(out)

bovw = BoVWextractor(K=K)
bovw.fit(X_train)
Xbovw_train = bovw.transform(X_train)

end = time.clock()
colorprint(Color.BLUE, 'Done! Elapsed time: ' + str(end - start) + 'sec\n')

colorprint(Color.BLUE, 'Start training the SVM...\n')
start = time.clock()

svm_classifier = svm.SVC()
svm_classifier.fit(Xbovw_train, y_train)

end = time.clock()
colorprint(Color.BLUE, 'Done! Elapsed time: ' + str(end - start) + 'sec\n')

# Test the system
colorprint(Color.BLUE, 'Start testing the full system...\n')
start = time.clock()

y_test = []
X_test = []
for cls in CLASSES:
    for imname in os.listdir(os.path.join(DATASET_DIR, 'test', cls)):
        im = Image.open(os.path.join(DATASET_DIR, 'test', cls, imname))
        y_test.append(cls)
        patches = extract_patches_2d(np.array(im), (PATCH_SIZE, PATCH_SIZE), max_patches=MAX_PATCHES)
        out = model_layer.predict(patches)
        X_test.append(out)

Xbovw_test = bovw.transform(X_test)

pred = svm_classifier.predict(Xbovw_test)

accuracy = accuracy_score(y_test, pred)

end = time.clock()
colorprint(Color.BLUE, 'Done! Elapsed time: ' + str(end - start) + 'sec\n')

print 'Accuracy on test: ' + str(100 * accuracy)
