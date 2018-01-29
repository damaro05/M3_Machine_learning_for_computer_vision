import os
import numpy as np
import random
from keras import optimizers
from definitive_model_rand_search import CNNmodel

os.environ["CUDA_VISIBLE_DEVICES"] = "4"

train_data_dir = '/share/datasets/MIT_split/train'
val_data_dir = '/share/datasets/MIT_split/test'
test_data_dir = '/share/datasets/MIT_split/test'

type = 2 #np.random.randint(1, 8)

batch_size_range = [32, 128]
epoch_range = [25, 50]
momentum_range = [0.0, 0.9]
optimizer_range = ['SGD', 'RMSprop', 'Adagrad', 'Adadelta', 'Adam', 'Adamax', 'Nadam']

batch_size = random.randint(batch_size_range[0],batch_size_range[1])
#number_of_epoch = random.randint(epoch_range[0],epoch_range[1])
number_of_epoch = 30
momentum = random.uniform(momentum_range[0],momentum_range[1])
optimizer_name = random.choice(optimizer_range)
#optimizer_name = 'Adadelta'
#dropout = random.randint(0,1)
dropout = 1
datageneration = random.randint(0,1)

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

cnn = CNNmodel(optimizer=optimizer, dropout=dropout, datagen=datageneration)
cnn.fit_directory(train_data_dir, batch_size=batch_size, epochs=number_of_epoch, val_path=val_data_dir)

results = cnn.evaluate(test_data_dir)
with open('log.csv', 'ab') as f:
    f.write(cnn.born_time + ';' + str(type) + ';' + cnn.identifier + ';' + str(results[0]) + ';' + str(results[1]) + ';' + str(batch_size) + ';' + str(number_of_epoch) + ';' + str(optimizer_name) + ';' + str(learn_rate) + ';' + str(momentum) + ';' + str(dropout) + ';' + str(datageneration) +'\n')

print 'Train: ' + str(cnn.evaluate(train_data_dir))
print 'Test: ' + str(cnn.evaluate(test_data_dir))
