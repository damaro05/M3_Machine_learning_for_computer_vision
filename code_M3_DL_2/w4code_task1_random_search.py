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
from keras.layers import Dense, GlobalAveragePooling2D, Dropout
from keras import backend as K
# from keras.utils.visualize_util import plot
from keras.preprocessing.image import ImageDataGenerator
import matplotlib.pyplot as plt
from keras.utils import plot_model
import random
from keras import optimizers

os.environ["CUDA_VISIBLE_DEVICES"] = "5"
train_data_dir = '/share/datasets/MIT_split/train'
val_data_dir = '/share/datasets/MIT_split/test'
test_data_dir = '/share/datasets/MIT_split/test'
img_width = 224
img_height = 224
batch_size = 64
n_train_samples = 400
number_of_epoch = 50
script_identifier = 'task_1'
plot_history = True

# Random search of parameters

batch_size_range = [32, 128]
#dropout_range = [0.25, 0.6]
momentum_range = [0.0, 0.9]
optimizer_range = ['SGD', 'RMSprop', 'Adagrad', 'Adadelta', 'Adam', 'Adamax', 'Nadam']

batch_size = random.randint(batch_size_range[0],batch_size_range[1])
#dropout_range = random.uniform(dropout_range[0],dropout_range[1])
momentum = random.uniform(momentum_range[0],momentum_range[1])
optimizer_name = random.choice(optimizer_range)

if (optimizer_name == 'SGD'):
    learn_rate_range = [0.005, 0.025]
    learn_rate = random.uniform(learn_rate_range[0],learn_rate_range[1])
    optimizer = optimizers.SGD(lr=learn_rate, momentum=momentum)
elif(optimizer_name == 'RMSprop'):
    learn_rate_range = [0.0005, 0.005]
    learn_rate = random.uniform(learn_rate_range[0],learn_rate_range[1])
    optimizer = optimizers.RMSprop(lr=learn_rate)
elif(optimizer_name == 'Adagrad'):
    learn_rate_range = [0.005, 0.025]
    learn_rate = random.uniform(learn_rate_range[0],learn_rate_range[1])
    optimizer = optimizers.Adagrad(lr=learn_rate)
elif(optimizer_name == 'Adadelta'):
    learn_rate_range = [0.9, 1.0]
    learn_rate = random.uniform(learn_rate_range[0],learn_rate_range[1])
    optimizer = optimizers.Adadelta(lr=learn_rate)
elif(optimizer_name == 'Adam'):
    learn_rate_range = [0.0005, 0.005]
    learn_rate = random.uniform(learn_rate_range[0],learn_rate_range[1])
    optimizer = optimizers.Adam(lr=learn_rate)
elif(optimizer_name == 'Adamax'):
    learn_rate_range = [0.0005, 0.005]
    learn_rate = random.uniform(learn_rate_range[0],learn_rate_range[1])
    optimizer = optimizers.Adamax(lr=learn_rate)
elif(optimizer_name == 'Nadam'):
    learn_rate_range = [0.0005, 0.005]
    learn_rate = random.uniform(learn_rate_range[0],learn_rate_range[1])
    optimizer = optimizers.Nadam(lr=learn_rate)

print 'Random parameters: batch=', batch_size, ' opt=', optimizer_name, ' learning rate=', learn_rate, ' momentum=', momentum



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

# New hybrid model
x = base_model.get_layer('block4_pool').output
x = Dropout(0.5)(x)
x = Conv2D(filters=512, kernel_size=(3, 3), strides=(2, 2))(x)
x = Dropout(0.5)(x)
x = Flatten()(x)
x = Dense(4096, activation='relu', name='fc1')(x)
x = Dense(8, activation='softmax', name='predictions')(x)
model = Model(inputs=base_model.input, outputs=x)

model_identifier = str(hash(str(model.get_config())))
print('Current model: '+model_identifier)
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
#estop_loss=EarlyStopping(monitor='val_loss',min_delta=.25,patience=5,verbose=1)
#estop_acc=EarlyStopping(monitor='val_acc',min_delta=.01,patience=5,verbose=1)
history = model.fit_generator(train_generator,
                              steps_per_epoch=n_train_samples / batch_size,  # batch_size*(int(400*1881/1881//batch_size)+1)
                              epochs=number_of_epoch,
                              validation_data=validation_generator,
                              validation_steps=validation_generator.samples / batch_size)
                              #callbacks=[estop_acc,estop_loss])
# save weights
model.save_weights(os.path.join('dump', 'models',
                                script_identifier + '_' + model_identifier + '_' + str(batch_size) + '_' + str(
                                    number_of_epoch) + '.h5'))
# save history
with open(os.path.join('dump', 'histories',
                       script_identifier + '_' + model_identifier + '_' + str(batch_size) + '_' + str(
                           number_of_epoch) + '_history.pklz'),
          'wb') as f:
    cPickle.dump(
        (history.epoch, history.history, history.params, history.validation_data, model.get_config()), f,
        cPickle.HIGHEST_PROTOCOL)

#evaluate
result = model.evaluate_generator(test_generator)
print result

# plot history graphs
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