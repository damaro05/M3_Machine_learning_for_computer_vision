import os
from definitive_model import CNNmodel

os.environ["CUDA_VISIBLE_DEVICES"] = "4"
train_data_dir = '/home/master04/data/MIT_split_val/train_split'
val_data_dir = '/home/master04/data/MIT_split_val/validation_split'
test_data_dir = '/share/datasets/MIT_split/test'
batch_size = 40
number_of_epoch = 120
cnn = CNNmodel()
cnn.fit_directory(train_data_dir, batch_size=batch_size, epochs=number_of_epoch, val_path=val_data_dir,
                  save_weights=True)
print 'Train: ' + str(cnn.evaluate(train_data_dir))
print 'Test: ' + str(cnn.evaluate(test_data_dir))
