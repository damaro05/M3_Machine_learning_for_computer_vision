from keras.preprocessing.image import ImageDataGenerator


def build_patch_generator(DATASET_DIR, PATCH_SIZE, BATCH_SIZE, SPLIT, CLASSES):
    if SPLIT == 'train':
        horizontal_flip = True
    elif SPLIT == 'test':
        horizontal_flip = False
    else:
        print('ERROR: SPLIT must be \'train\' or \'test\'')
        exit(-1)

    datagen = ImageDataGenerator(
        rescale=1. / 255,
        horizontal_flip=horizontal_flip)

    # this is a generator that will read pictures found in
    # subfolers of 'data/SPLIT', and indefinitely generate
    # batches of augmented image data
    generator = datagen.flow_from_directory(
        DATASET_DIR + '/' + SPLIT,  # this is the target directory
        target_size=(PATCH_SIZE, PATCH_SIZE),  # all images will be resized to IMG_SIZExIMG_SIZE
        batch_size=BATCH_SIZE,
        classes=CLASSES,
        class_mode='categorical')  # since we use binary_crossentropy loss, we need categorical labels
    return generator
