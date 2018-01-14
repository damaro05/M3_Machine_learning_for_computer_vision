import matplotlib

matplotlib.use('Agg')
import time

import cPickle

import numpy as np
from skimage.io import imread
from sklearn.feature_extraction.image import extract_patches_2d
from sklearn.metrics import accuracy_score

from patches.patch_data_loader import build_patch_generator
from patches.patch_mlp_builder import build_patch_mlp
from utils import colorprint, Color, softmax, generate_image_patches_db
import os

## PARAMETERS ##########################################################################################################
PATCH_SIZE = 64
BATCH_SIZE = 16
EPOCHS = 150
MAX_PATCHES = 64
DATASET_DIR = '/share/datasets/MIT_split'
PATCHES_DIR = '/Users/leki/Code/Module 3/Databases/MIT_split_patches_64'#'/home/master04/data/MIT_split_patches_16'
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
    generate_image_patches_db(DATASET_DIR, PATCHES_DIR, patch_size=PATCH_SIZE,max_patches=MAX_PATCHES)
    colorprint(Color.BLUE, 'Done!\n')

# Build the NN
model = build_patch_mlp(PATCH_SIZE)

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
        steps_per_epoch=10,#train_generator.samples // BATCH_SIZE,
        epochs=EPOCHS,
        validation_data=validation_generator,
        validation_steps=10)#validation_steps)

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
    # plot_history(history, model_identifier, metric='acc', plot_validation=TRAIN_WITH_VALIDATION, path='dump/patch_histories/')
    # summarize history for loss
    # plot_history(history, model_identifier, metric='loss', plot_validation=TRAIN_WITH_VALIDATION, path='dump/patch_histories/')

# Test the model
colorprint(Color.BLUE, 'Start testing...\n')
start = time.clock()

test_model = build_patch_mlp(PATCH_SIZE, phase='TEST')
test_model.load_weights('dump/patch_models/' + model_identifier + '.h5')

CLASSES = np.array(CLASSES)

y = []
pred = []
for cls in CLASSES:
    for imname in os.listdir(os.path.join(DATASET_DIR, 'test', cls)):
        im = imread(os.path.join(DATASET_DIR, 'test', cls, imname))
        y.append(cls)
        patches = extract_patches_2d(im, (PATCH_SIZE, PATCH_SIZE), max_patches=MAX_PATCHES)
        out = test_model.predict(patches)
        predicted_cls = np.argmax(softmax(np.mean(out, axis=0)))
        pred.append(CLASSES[predicted_cls])

accuracy = accuracy_score(y, pred)
end = time.clock()
colorprint(Color.BLUE, 'Done! Elapsed time: ' + str(end - start) + 'sec\n')

print 'Accuracy on test: ' + str(100 * accuracy)
