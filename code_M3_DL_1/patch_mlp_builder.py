import os
from keras import Sequential
from keras.layers import Reshape, Dense
from keras.utils import plot_model

from utils import colorprint, Color


def build_patch_mlp(PATCH_SIZE,phase='TRAIN'):
    colorprint(Color.BLUE, 'Building MLP model...\n')

    model = Sequential()
    model.add(Reshape((PATCH_SIZE * PATCH_SIZE * 3,), input_shape=(PATCH_SIZE, PATCH_SIZE, 3)))
    model.add(Dense(units=2048, activation='relu'))
    # model.add(Dense(units=1024, activation='relu'))
    if phase.capitalize() == 'TEST':
        model.add(Dense(units=8, activation='linear'))  # In test phase we softmax the average output over the image patches
    else:
        model.add(Dense(units=8, activation='softmax'))

    model.compile(loss='categorical_crossentropy',
                  optimizer='sgd',
                  metrics=['accuracy'])

    model.summary()
    if not os.path.exists('patch_models'):
        os.mkdir('patch_models')
    plot_model(model, to_file='patch_models/' + str(hash(str(model.get_config()))) + '.png', show_shapes=True,
               show_layer_names=True)

    colorprint(Color.BLUE, 'Done!\n')
    return model
