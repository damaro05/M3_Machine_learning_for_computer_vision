from utils import *
from keras.models import Sequential
from keras.layers import Flatten, Dense, Reshape, Conv2D, MaxPooling2D, Dropout
from keras.preprocessing.image import ImageDataGenerator
from keras.utils import plot_model

import matplotlib.pyplot as plt
#user defined variables
IMG_SIZE    = 32
BATCH_SIZE  = 16
DATASET_DIR = '/Users/damaro/Desktop/M.Computer Vision/M3.Machine learning/Practices/Databases/MIT_split'
MODEL_FNAME = 'TestingNetwork.h5'

EPOCHS = 50
input_shape = (IMG_SIZE, IMG_SIZE, 3)

if not os.path.exists(DATASET_DIR):
  colorprint(Color.RED, 'ERROR: dataset directory '+DATASET_DIR+' do not exists!\n')
  quit()


colorprint(Color.BLUE, 'Building MLP model...\n')

#Build the Multi Layer Perceptron model
model = Sequential()
model.add(Conv2D(32, (3,3), padding='same', activation='relu', input_shape=input_shape))
model.add(Conv2D(32, (3,3), activation='relu'))
model.add(MaxPooling2D( pool_size=(2, 2)))
model.add(Dropout(0.25))

model.add(Conv2D(64, (3,3), padding='same', activation='relu'))
model.add(Conv2D(64, (3,3), activation='relu'))
model.add(MaxPooling2D( pool_size=(2, 2)))
model.add(Dropout(0.25))

model.add(Conv2D(64, (3, 3), padding='same', activation='relu'))
model.add(Conv2D(64, (3, 3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.25))

model.add(Flatten())
model.add(Dense(512, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(8, activation='softmax'))


#model.add(Reshape((IMG_SIZE*IMG_SIZE*3,),input_shape=(IMG_SIZE, IMG_SIZE, 3),name='first'))
#model.add(Dense(units=2048, activation='relu',name='second'))
#model.add(Dense(units=1024, activation='relu'))
#model.add(Dense(units=8, activation='softmax'))

model.compile(loss='categorical_crossentropy',
              optimizer='rmsprop',
              metrics=['accuracy'])

print(model.summary())
plot_model(model, to_file='modelTestingNetwork.png', show_shapes=True, show_layer_names=True)
colorprint(Color.BLUE, 'Done!\n')
if os.path.exists(MODEL_FNAME):
  colorprint(Color.YELLOW, 'WARNING: model file '+MODEL_FNAME+' exists and will be overwritten!\n')
colorprint(Color.BLUE, 'Start training...\n')

# this is the dataset configuration we will use for training
# only rescaling
train_datagen = ImageDataGenerator(
        rescale=1./255,
        width_shift_range=0.1,
        height_shift_range=0.1,
        vertical_flip=False,
        horizontal_flip=True)

# this is the dataset configuration we will use for testing:
# only rescaling
test_datagen = ImageDataGenerator(rescale=1./255)

# this is a generator that will read pictures found in
# subfolers of 'data/train', and indefinitely generate
# batches of augmented image data
train_generator = train_datagen.flow_from_directory(
        DATASET_DIR+'/train',  # this is the target directory
        target_size=(IMG_SIZE, IMG_SIZE),  # all images will be resized to IMG_SIZExIMG_SIZE
        batch_size=BATCH_SIZE,
        classes = ['coast','forest','highway','inside_city','mountain','Opencountry','street','tallbuilding'],
        class_mode='categorical')  # since we use binary_crossentropy loss, we need categorical labels

# this is a similar generator, for validation data
validation_generator = test_datagen.flow_from_directory(
        DATASET_DIR+'/test',
        target_size=(IMG_SIZE, IMG_SIZE),
        batch_size=BATCH_SIZE,
        classes = ['coast','forest','highway','inside_city','mountain','Opencountry','street','tallbuilding'],
        class_mode='categorical')

history = model.fit_generator(
        train_generator,
        steps_per_epoch=1881 // BATCH_SIZE, #should be 1881 full dataset - 120 small dataset
        epochs=EPOCHS,
        validation_data=validation_generator,
        validation_steps=807 // BATCH_SIZE) #should be 807 to full dataset - 120 small dataset

colorprint(Color.BLUE, 'Done!\n')
colorprint(Color.BLUE, 'Saving the model into '+MODEL_FNAME+' \n')
model.save_weights(MODEL_FNAME)  # always save your weights after training or during training
colorprint(Color.BLUE, 'Done!\n')

  # summarize history for accuracy
plt.plot(history.history['acc'])
plt.plot(history.history['val_acc'])
plt.title('model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'validation'], loc='upper left')
plt.savefig('16BATCH_SIZEaccuracyTesting.jpg')
plt.close()
  # summarize history for loss
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'validation'], loc='upper left')
plt.savefig('16BATCH_SIZElossTesting.jpg')

#Evaluate the model
result = model.evaluate_generator(validation_generator)
print('Test accuracy: ' + str(result[1]*100) + '%')
print('Test loss: ' + str(result[0]))