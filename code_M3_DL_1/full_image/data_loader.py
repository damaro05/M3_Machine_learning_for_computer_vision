import numpy as np
import os
from keras.preprocessing.image import ImageDataGenerator
from skimage.io import imread
from skimage.transform import resize



def build_generator(DATASET_DIR, IMG_SIZE, BATCH_SIZE, SPLIT,CLASSES):
    if SPLIT == 'train':
        horizontal_flip = True
        rotation_range = 0#40
        width_shift_range = 0.1
        height_shift_range = 0.1
        zoom_range=0.1
        shuffle=True
    elif SPLIT == 'test':
        horizontal_flip = False
        rotation_range = 0
        width_shift_range = 0
        height_shift_range = 0
        zoom_range = 0
        shuffle=False
    else:
        print('ERROR: SPLIT must be \'train\' or \'test\'')
        exit(-1)

    datagen = ImageDataGenerator(
        rescale=1. / 255,
        horizontal_flip=horizontal_flip,
        rotation_range=rotation_range,
        width_shift_range=width_shift_range,
        height_shift_range=height_shift_range,
        zoom_range=zoom_range)

    # this is a generator that will read pictures found in
    # subfolers of 'data/SPLIT', and indefinitely generate
    # batches of augmented image data
    generator = datagen.flow_from_directory(
        DATASET_DIR + '/' + SPLIT,  # this is the target directory
        target_size=(IMG_SIZE, IMG_SIZE),  # all images will be resized to IMG_SIZExIMG_SIZE
        batch_size=BATCH_SIZE,
        classes=CLASSES,
        class_mode='categorical',
        shuffle=shuffle)  # since we use binary_crossentropy loss, we need categorical labels
    return generator

def load_dataset(DATASET_DIR, SPLIT, IMG_RESIZE=None, classes=None):
    if classes==None:
        classes = os.listdir(os.path.join(DATASET_DIR, SPLIT))
    y = []
    X = []
    for cls in classes:
        for imname in os.listdir(os.path.join(DATASET_DIR, SPLIT, cls)):
            im = imread(os.path.join(DATASET_DIR, SPLIT, cls, imname))
            y.append(cls)
            if IMG_RESIZE!= None:
                im = resize(im, (IMG_RESIZE[0], IMG_RESIZE[1]),mode='reflect')
            X.append(im)
    return np.array(X),np.array(y)