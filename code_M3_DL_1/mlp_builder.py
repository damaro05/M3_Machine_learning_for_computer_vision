import os
from keras import Sequential
from keras.layers import Reshape, Dense
from keras.utils import plot_model

from utils import colorprint, Color


def build_mlp(IMG_SIZE):
    colorprint(Color.BLUE, 'Building MLP model...\n')

    # Build the Multi Layer Perceptron model
    model = Sequential()
    model.add(Reshape((IMG_SIZE * IMG_SIZE * 3,), input_shape=(IMG_SIZE, IMG_SIZE, 3), name='first'))
    model.add(Dense(units=2048, activation='relu', name='second'))
    #model.add(Dense(units=512, activation='relu', name='third'))
    model.add(Dense(units=8, activation='softmax'))

    model.compile(loss='categorical_crossentropy',
                  optimizer='sgd',
                  metrics=['accuracy'])

    model.summary()
    if not os.path.exists('models'):
        os.mkdir('models')
    plot_model(model, to_file='models/' + str(hash(str(model.get_config()))) + '.png', show_shapes=True,
               show_layer_names=True)

    colorprint(Color.BLUE, 'Done!\n')
    return model

def build_mlp_svm(IMG_SIZE):
    colorprint(Color.BLUE, 'Building MLP model...\n')

    # Build the Multi Layer Perceptron model
    model = Sequential()
    model.add(Reshape((IMG_SIZE * IMG_SIZE * 3,), input_shape=(IMG_SIZE, IMG_SIZE, 3), name='first'))
    model.add(Dense(units=128, activation='relu', name='second'))
    model.add(Dense(units=512, activation='relu', name='third'))
    model.add(Dense(units=8, activation='softmax'))

    model.compile(loss='categorical_crossentropy',
                  optimizer='sgd',
                  metrics=['accuracy'])

    model.summary()
    if not os.path.exists('models'):
        os.mkdir('models')
    plot_model(model, to_file='models/' + str(hash(str(model.get_config()))) + '.png', show_shapes=True,
               show_layer_names=True)

    colorprint(Color.BLUE, 'Done!\n')
    return model
