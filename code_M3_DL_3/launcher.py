import os
from definitive_model import CNNmodel

os.environ["CUDA_VISIBLE_DEVICES"] = "4"
train_data_dir = '/share/datasets/MIT_split/train'
val_data_dir = '/share/datasets/MIT_split/test'
test_data_dir = '/share/datasets/MIT_split/test'
batch_size = 32
number_of_epoch = 25
cnn = CNNmodel()
cnn.fit_directory(train_data_dir, batch_size=batch_size, epochs=number_of_epoch, val_path=val_data_dir)
print 'Train: ' + str(cnn.evaluate(train_data_dir))
print 'Test: ' + str(cnn.evaluate(test_data_dir))
