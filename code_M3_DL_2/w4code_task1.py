import matplotlib
from keras.callbacks import EarlyStopping
from keras.initializers import glorot_uniform

matplotlib.use('Agg')
import cPickle
import os
from keras.applications.vgg16 import VGG16
from keras.preprocessing import image
from keras.models import Model
from keras.layers import Flatten, AveragePooling2D, MaxPooling2D, Conv2D
from keras.layers import Dense, GlobalAveragePooling2D
from keras import backend as K
# from keras.utils.visualize_util import plot
from keras.preprocessing.image import ImageDataGenerator
import matplotlib.pyplot as plt
from keras.utils import plot_model

os.environ["CUDA_VISIBLE_DEVICES"] = "4"
train_data_dir = '/share/datasets/MIT_split/train'
val_data_dir = '/share/datasets/MIT_split/test'
test_data_dir = '/share/datasets/MIT_split/test'
img_width = 224
img_height = 224
batch_size = 64
n_train_samples = 800
number_of_epoch = 200
script_identifier = 'task_1'
plot_history = True


def preprocess_input(x, dim_ordering='default'):
    if dim_ordering == 'default':
        dim_ordering = K.image_dim_ordering()
    assert dim_ordering in {'tf', 'th'}

    if dim_ordering == 'th':
        # 'RGB'->'BGR'
        x = x[::-1, :, :]
        # Zero-center by mean pixel
        x[0, :, :] -= 109.07621812  # 103.939
        x[1, :, :] -= 115.45609435  # 116.779
        x[2, :, :] -= 114.70990406  # 123.68
        # Normalize by std
        x[0, :, :] /= 56.91689916
        x[1, :, :] /= 55.4694083
        x[2, :, :] /= 59.14847488
    else:
        # 'RGB'->'BGR'
        x = x[:, :, ::-1]
        # Zero-center by mean pixel
        x[:, :, 0] -= 109.07621812  # 103.939
        x[:, :, 1] -= 115.45609435  # 116.779
        x[:, :, 2] -= 114.70990406  # 123.68
        # Normalize by std
        x[:, :, 0] /= 56.91689916
        x[:, :, 1] /= 55.4694083
        x[:, :, 2] /= 59.14847488
    return x

if not os.path.exists(os.path.join('dump', 'models')):
    os.makedirs(os.path.join('dump', 'models'))

if not os.path.exists(os.path.join('dump', 'histories')):
    os.makedirs(os.path.join('dump', 'histories'))

# create the base pre-trained model
base_model = VGG16(weights='imagenet')
plot_model(base_model, to_file=os.path.join('dump', 'models', 'modelVGG16.png'), show_shapes=True,
           show_layer_names=True)

# Not valid: mem alloc error
x = base_model.get_layer('block4_pool').output
x = Conv2D(filters=512, kernel_size=(3, 3), strides=(2, 2))(x)

# x = base_model.get_layer('block4_conv3').output
# Method 1: 7x7x512 maybe too difficult? doesn't work properly
# Try with maxpool
# See what ramon says in the email
# x = MaxPooling2D(pool_size=(4, 4), strides=(4, 4))(x)  # Strides? Size of pixel jump
x = Flatten()(x)

# Method 2: maybe too much? only one value per channel i.e. 1x512
# x = GlobalAveragePooling2D()(x)

x = Dense(4096, activation='relu', name='fc1')(x)
# x = Dense(4096, activation='relu', name='fc2')(x)
x = Dense(8, activation='softmax', name='predictions')(x)

model = Model(inputs=base_model.input, outputs=x)
model_identifier = str(hash(str(model.get_config())))
plot_model(model, to_file=os.path.join('dump', 'models', script_identifier + '_' + model_identifier + '.png'),
           show_shapes=True, show_layer_names=True)
for layer in base_model.layers:
    layer.trainable = False

model.compile(loss='categorical_crossentropy', optimizer='adadelta', metrics=['accuracy'])
for layer in model.layers:
    print layer.name, layer.trainable

datagen = ImageDataGenerator(preprocessing_function=preprocess_input,
                             rotation_range=0.,
                             width_shift_range=0.,
                             height_shift_range=0.,
                             shear_range=0.,
                             zoom_range=0.,
                             channel_shift_range=0.,
                             fill_mode='nearest',
                             cval=0.,
                             horizontal_flip=False,
                             vertical_flip=False)

train_generator = datagen.flow_from_directory(train_data_dir,
                                              target_size=(img_width, img_height),
                                              batch_size=batch_size,
                                              class_mode='categorical')

test_generator = datagen.flow_from_directory(test_data_dir,
                                             target_size=(img_width, img_height),
                                             batch_size=batch_size,
                                             class_mode='categorical',
                                             shuffle=False)

validation_generator = datagen.flow_from_directory(val_data_dir,
                                                   target_size=(img_width, img_height),
                                                   batch_size=batch_size,
                                                   class_mode='categorical')
#first training
estop_loss=EarlyStopping(monitor='val_loss',min_delta=.25,patience=5,verbose=1)
estop_acc=EarlyStopping(monitor='val_acc',min_delta=.01,patience=5,verbose=1)
history = model.fit_generator(train_generator,
                              steps_per_epoch=n_train_samples / batch_size,  # batch_size*(int(400*1881/1881//batch_size)+1)
                              epochs=number_of_epoch,
                              validation_data=validation_generator,
                              validation_steps=validation_generator.samples / batch_size,
                              callbacks=[estop_acc,estop_loss])
model.save_weights(os.path.join('dump', 'models',
                                script_identifier + '_' + model_identifier + '_' + str(batch_size) + '_' + str(
                                    number_of_epoch) + '.h5'))

with open(os.path.join('dump', 'histories',
                       script_identifier + '_' + model_identifier + '_' + str(batch_size) + '_' + str(
                           number_of_epoch) + '_history.pklz'),
          'wb') as f:
    cPickle.dump(
        (history.epoch, history.history, history.params, history.validation_data, model.get_config()), f,
        cPickle.HIGHEST_PROTOCOL)

#second training
for layer in base_model.layers:
    layer.trainable = True

model.compile(loss='categorical_crossentropy', optimizer='adadelta', metrics=['accuracy'])
for layer in model.layers:
    print layer.name, layer.trainable
model.load_weights(os.path.join('dump', 'models',
                                script_identifier + '_' + model_identifier + '_' + str(batch_size) + '_' + str(
                                    number_of_epoch) + '.h5'))
history2 = model.fit_generator(train_generator,
                              steps_per_epoch=n_train_samples / batch_size,  # batch_size*(int(400*1881/1881//batch_size)+1)
                              epochs=number_of_epoch,
                              validation_data=validation_generator,
                              validation_steps=validation_generator.samples / batch_size,
                              callbacks=[estop_acc,estop_loss])
model.save_weights(os.path.join('dump', 'models',
                                script_identifier + '_' + model_identifier + '_' + str(batch_size) + '_' + str(
                                    number_of_epoch) + '.h5'))

with open(os.path.join('dump', 'histories',
                       script_identifier + '_' + model_identifier + '_' + str(batch_size) + '_' + str(
                           number_of_epoch) + '_history2.pklz'),
          'wb') as f:
    cPickle.dump(
        (history2.epoch, history2.history, history2.params, history2.validation_data, model.get_config()), f,
        cPickle.HIGHEST_PROTOCOL)

#evaluate
result = model.evaluate_generator(test_generator)
print result

# list all data in history

if plot_history:
    # summarize history for accuracy
    plt.plot(history.history['acc'])
    plt.plot(history.history['val_acc'])
    plt.title('model accuracy')
    plt.ylabel('accuracy')
    plt.xlabel('epoch')
    plt.legend(['train', 'validation'], loc='upper left')
    plt.savefig(os.path.join('dump', 'histories',
                             script_identifier + '_' + model_identifier + '_' + str(batch_size) + '_' + str(
                                 number_of_epoch) + '_accuracy.jpg'))
    plt.close()
    # summarize history for loss
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title('model loss')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend(['train', 'validation'], loc='upper left')
    plt.savefig(os.path.join('dump', 'histories',
                             script_identifier + '_' + model_identifier + '_' + str(batch_size) + '_' + str(
                                 number_of_epoch) + '_loss.jpg'))