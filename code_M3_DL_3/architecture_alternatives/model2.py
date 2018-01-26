import cPickle
import os
import time
from keras import Sequential
from keras import backend as K
from keras.layers import Conv2D, Flatten, Dense, MaxPooling2D
from keras.optimizers import Adam
from keras.preprocessing.image import ImageDataGenerator
from keras.utils import plot_model
import utils
import numpy as np


class CNNmodel2:
    def __init__(self, img_size=(256, 256), dump_path='dump/'):
        # Random parameters
        conv1_filters = np.random.randint(1,65) #65
        conv2_filters = np.random.randint(1,65) #65
        conv1_kernel = np.random.randint(2,12) #10
        conv2_kernel = np.random.randint(2,12) #10
        conv1_strides = np.random.randint(1,conv1_kernel/2+1)
        conv2_strides = np.random.randint(1,conv2_kernel/2+1)
        maxpool1_size = np.random.randint(2,12) #10
        maxpool2_size = np.random.randint(2,12) #10
        fc1_units = 2**np.random.randint(6,12) #11
        fc2_units = 2**np.random.randint(6,12) #11


        # Model architecture
        self.model = Sequential()
        self.model.add(Conv2D(filters=conv1_filters,
                              kernel_size=(conv1_kernel, conv1_kernel),
                              strides=(conv1_strides,conv1_strides),
                              activation='relu',
                              input_shape=(img_size[0], img_size[1], 3),
                              name='conv1'))
        self.model.add(MaxPooling2D(pool_size=(maxpool1_size, maxpool1_size),
                                    strides=None,
                                    name='maxpool1'))
        self.model.add(Conv2D(filters=conv2_filters,
                              kernel_size=(conv2_kernel, conv2_kernel),
                              strides=(conv2_strides, conv2_strides),
                              activation='relu',
                              name='conv2'))
        self.model.add(MaxPooling2D(pool_size=(maxpool2_size, maxpool2_size),
                                    strides=None,
                                    name='maxpool2'))
        self.model.add(Flatten())
        self.model.add(Dense(units=fc1_units, activation='relu', name='fc1'))
        self.model.add(Dense(units=fc2_units, activation='relu', name='fc2'))
        self.model.add(Dense(units=8, activation='softmax', name='classif'))

        # Optimizer
        optimizer = Adam()

        # Compile
        self.model.compile(loss='categorical_crossentropy',
                           optimizer=optimizer,
                           metrics=['accuracy'])
        # Parameters
        self.born_time = time.strftime('%Y%m%d%H%M%S',time.gmtime())
        self.identifier = str(hash(str(self.model.get_config())))
        self.dump_path = os.path.join(dump_path, str(self.born_time) + '_' + self.identifier)
        self.input_img_size = img_size

        # Print
        if not os.path.exists(self.dump_path):
            os.makedirs(self.dump_path)
        self.model.summary()
        print('Current model: ' + self.identifier)
        plot_model(self.model, show_shapes=True, show_layer_names=True,
                   to_file=os.path.join(self.dump_path, self.identifier + '.png'))

    def _train_generator(self, path, batch_size):
        datagen = ImageDataGenerator(preprocessing_function=self._preprocess_input,
                                     rotation_range=0,
                                     width_shift_range=0.,
                                     height_shift_range=0.,
                                     shear_range=0.,
                                     zoom_range=0.,
                                     channel_shift_range=0.,
                                     fill_mode='reflect',
                                     cval=0.,
                                     horizontal_flip=False,
                                     vertical_flip=False)
        return datagen.flow_from_directory(path,
                                           target_size=self.input_img_size,
                                           batch_size=batch_size,
                                           class_mode='categorical')

    def _test_val_generator(self, path, batch_size):
        datagen = ImageDataGenerator(preprocessing_function=self._preprocess_input)
        return datagen.flow_from_directory(path,
                                           target_size=self.input_img_size,
                                           batch_size=batch_size,
                                           class_mode='categorical',
                                           shuffle=False)

    def fit_directory(self, path, batch_size, epochs, val_path=None, save_weights=False):
        train_generator = self._train_generator(path, batch_size)
        if val_path is None:
            validation_generator = None
            validation_steps = None
        else:
            validation_generator = self._test_val_generator(val_path, batch_size)
            validation_steps = validation_generator.samples / batch_size

        history = self.model.fit_generator(train_generator,
                                           steps_per_epoch=train_generator.samples / batch_size,
                                           epochs=epochs,
                                           validation_data=validation_generator,
                                           validation_steps=validation_steps)
        utils.plot_history(history, self.dump_path, identifier='e' + str(epochs) + '_b' + str(batch_size))
        with open(os.path.join(self.dump_path, 'e' + str(epochs) + '_b' + str(batch_size) + '_history.pklz'),
                  'wb') as f:
            cPickle.dump(
                (history.epoch, history.history, history.params, history.validation_data, self.model.get_config()), f,
                cPickle.HIGHEST_PROTOCOL)
        if save_weights:
            self.model.save_weights(
                os.path.join(self.dump_path, 'e' + str(epochs) + '_b' + str(batch_size) + '_weights.h5'))
        return history

    def evaluate(self, path):
        test_generator = self._test_val_generator(path, batch_size=32)
        return self.model.evaluate_generator(test_generator)

    def _preprocess_input(self, x, dim_ordering='default'):
        if dim_ordering == 'default':
            dim_ordering = K.image_dim_ordering()
        assert dim_ordering in {'tf', 'th'}

        mean = [109.07621812, 115.45609435, 114.70990406]
        std = [56.91689916, 55.4694083, 59.14847488]
        if dim_ordering == 'th':
            # Zero-center by mean pixel
            x[0, :, :] -= mean[0]
            x[1, :, :] -= mean[1]
            x[2, :, :] -= mean[2]
            # Normalize by std
            x[0, :, :] /= std[0]
            x[1, :, :] /= std[1]
            x[2, :, :] /= std[2]
        else:
            # Zero-center by mean pixel
            x[:, :, 0] -= mean[0]
            x[:, :, 1] -= mean[1]
            x[:, :, 2] -= mean[2]
            # Normalize by std
            x[:, :, 0] /= std[0]
            x[:, :, 1] /= std[1]
            x[:, :, 2] /= std[2]
        return x
