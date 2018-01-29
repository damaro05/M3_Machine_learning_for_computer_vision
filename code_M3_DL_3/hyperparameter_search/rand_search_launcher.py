import os
import numpy as np
import random
from keras import optimizers
from definitive_model_rand_search import CNNmodel

os.environ["CUDA_VISIBLE_DEVICES"] = "4"

train_data_dir = '/home/master04/data/MIT_split_val/train_split'
val_data_dir = '/home/master04/data/MIT_split_val/validation_split'
test_data_dir = '/share/datasets/MIT_split/test'
batch_size = 8 * np.random.randint(4, 10)  # 64
number_of_epoch = 80

type = 2  # np.random.randint(1, 8)
optimizer_range = ['Adadelta', 'Adam', 'Adamax', 'Nadam']

optimizer_name = random.choice(optimizer_range)
dropout = np.random.rand(3) > 0.5
datageneration = 1
lr_reduce_factor = 1 / np.random.uniform(2, 10)

momentum = None
if (optimizer_name == 'Adadelta'):
    learn_rate_range = [0.5, 1.5]
    learn_rate = random.uniform(learn_rate_range[0], learn_rate_range[1])
    optimizer = optimizers.Adadelta(lr=learn_rate)
elif (optimizer_name == 'Adam'):
    learn_rate_range = [0.0005, 0.005]
    learn_rate = random.uniform(learn_rate_range[0], learn_rate_range[1])
    optimizer = optimizers.Adam(lr=learn_rate)
elif (optimizer_name == 'Adamax'):
    learn_rate_range = [0.0005, 0.005]
    learn_rate = random.uniform(learn_rate_range[0], learn_rate_range[1])
    optimizer = optimizers.Adamax(lr=learn_rate)
elif (optimizer_name == 'Nadam'):
    learn_rate_range = [0.0005, 0.005]
    learn_rate = random.uniform(learn_rate_range[0], learn_rate_range[1])
    optimizer = optimizers.Nadam(lr=learn_rate)

print 'Random parameters: batch=', batch_size, ' opt=', optimizer_name, ' learning rate=', learn_rate, ' dropout=', dropout, ' lr_reduce_factor=',lr_reduce_factor

cnn = CNNmodel(optimizer=optimizer, dropout=dropout, datagen=datageneration, lr_reduce_factor=lr_reduce_factor)
cnn.fit_directory(train_data_dir, batch_size=batch_size, epochs=number_of_epoch, val_path=val_data_dir)

test_results = cnn.evaluate(test_data_dir)
train_results = cnn.evaluate(train_data_dir)

with open('log.csv', 'ab') as f:
    f.write(cnn.born_time + ';' + str(type) + ';' + cnn.identifier + ';' + str(train_results[0]) + ';' + str(
        train_results[1]) + ';' + str(test_results[0]) + ';' + str(test_results[1]) + ';' + str(batch_size) + ';' + str(
        number_of_epoch) + ';' + str(optimizer_name) + ';' + str(learn_rate) + ';' + str(
        dropout) + ';' + str(datageneration) + ';' + str(lr_reduce_factor) + '\n')

print 'Train: ' + str(train_results)
print 'Test: ' + str(test_results)
