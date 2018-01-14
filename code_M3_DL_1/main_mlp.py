import matplotlib
from keras.preprocessing.image import ImageDataGenerator

matplotlib.use('Agg')
import time

import cPickle

import numpy as np
from sklearn.metrics import accuracy_score

from utils import colorprint, Color, plot_history
import os
from mlp_builder import build_mlp
from data_loader import build_generator, load_dataset

## PARAMETERS ##########################################################################################################
IMG_SIZE = 32
BATCH_SIZE = 32
EPOCHS = 150
DATASET_DIR = '/share/datasets/MIT_split'
CLASSES = ['coast', 'forest', 'highway', 'inside_city', 'mountain', 'Opencountry', 'street', 'tallbuilding']
RECOMPUTE = False
TRAIN_WITH_VALIDATION = True
########################################################################################################################

# Check the location of the dataset
if not os.path.exists(DATASET_DIR):
    colorprint(Color.RED, 'ERROR: dataset directory ' + DATASET_DIR + ' do not exists!\n')
    quit()

# Build the NN
model = build_mlp(IMG_SIZE)

# Train or load the weights
model_identifier = str(hash(str(model.get_config()))) + '_' + str(IMG_SIZE) + '_' + '_' + str(
    BATCH_SIZE) + '_' + '_' + str(EPOCHS)
if os.path.exists('models/' + model_identifier + '.h5') and not RECOMPUTE:
    model.load_weights('models/' + model_identifier + '.h5')
    colorprint(Color.GREEN, 'Loading model file models/' + model_identifier + '.h5\n')
else:
    if not os.path.exists('models'):
        os.mkdir('models')
    if os.path.exists(model_identifier + '.h5'):
        colorprint(Color.YELLOW,
                   'WARNING: model file models/' + model_identifier + '.h5 exists and will be overwritten!\n')

    # this is a generator that will read pictures found in
    # subfolers of 'data/train', and indefinitely generate
    # batches of augmented image data
    train_generator = build_generator(DATASET_DIR, IMG_SIZE, BATCH_SIZE, SPLIT='train', CLASSES=CLASSES)

    # this is a similar generator, for validation data
    if TRAIN_WITH_VALIDATION:
        validation_generator = build_generator(DATASET_DIR, IMG_SIZE, BATCH_SIZE, SPLIT='test', CLASSES=CLASSES)
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
        'models/' + model_identifier + '.h5')  # always save your weights after training or during training
    colorprint(Color.BLUE, 'Done!\n')
    if not os.path.exists('histories'):
        os.mkdir('histories')
    with open('histories/' + model_identifier + '_history.pklz', 'wb') as f:
        cPickle.dump(
            (history.epoch, history.history, history.params, history.validation_data, model.get_config()), f,
            cPickle.HIGHEST_PROTOCOL)

    # summarize history for accuracy
    plot_history(history, model_identifier, metric='acc', plot_validation=TRAIN_WITH_VALIDATION, path='histories')
    # summarize history for loss
    plot_history(history, model_identifier, metric='loss', plot_validation=TRAIN_WITH_VALIDATION, path='histories')

    print 'Accuracy on train: ' + str(100 * history.history['acc'][-1])
    if TRAIN_WITH_VALIDATION:
        print 'Accuracy on test: ' + str(100 * history.history['val_acc'][-1])

# Test the model
# test_generator = build_generator(DATASET_DIR, IMG_SIZE, BATCH_SIZE=1, SPLIT='test')

colorprint(Color.BLUE, 'Start testing...\n')
start = time.clock()
'''
X, y = load_dataset(DATASET_DIR, SPLIT='test', IMG_RESIZE=(IMG_SIZE, IMG_SIZE), classes=CLASSES)

pred = np.array(CLASSES)[np.argmax(model.predict(np.array(X)), axis=1)]
accuracy = accuracy_score(y,pred)

'''
test_generator = build_generator(DATASET_DIR, IMG_SIZE, BATCH_SIZE, SPLIT='test', CLASSES=CLASSES)
y = np.array(CLASSES)[test_generator.classes]

pred = np.array(CLASSES)[np.argmax(model.predict_generator(test_generator), axis=1)]
accuracy = accuracy_score(y,pred)
#accuracy = model.evaluate_generator(test_generator)[model.metrics_names.index('acc')]

end = time.clock()
colorprint(Color.BLUE, 'Done! Elapsed time: ' + str(end - start) + 'sec\n')

print 'Accuracy on test: ' + str(100 * accuracy)
