import os
import numpy as np
from architecture_alternatives.model1 import CNNmodel1
from architecture_alternatives.model2 import CNNmodel2
from architecture_alternatives.model3 import CNNmodel3
from architecture_alternatives.model4 import CNNmodel4
from architecture_alternatives.model5 import CNNmodel5
from architecture_alternatives.model6 import CNNmodel6
from architecture_alternatives.model7 import CNNmodel7

os.environ["CUDA_VISIBLE_DEVICES"] = "4"
train_data_dir = '/share/datasets/MIT_split/train'
val_data_dir = '/share/datasets/MIT_split/test'
test_data_dir = '/share/datasets/MIT_split/test'
batch_size = 32
number_of_epoch = 25
type = 2 #np.random.randint(1, 8)

if type == 1:
    cnn = CNNmodel1()
elif type == 2:
    cnn = CNNmodel2()
elif type == 3:
    cnn = CNNmodel3()
elif type == 4:
    cnn = CNNmodel4()
elif type == 5:
    cnn = CNNmodel5()
elif type == 6:
    cnn = CNNmodel6()
elif type == 7:
    cnn = CNNmodel7()
cnn.fit_directory(train_data_dir, batch_size=batch_size, epochs=number_of_epoch, val_path=val_data_dir)
results = cnn.evaluate(test_data_dir)
with open('log.csv', 'ab') as f:
    f.write(cnn.born_time + ';' + str(type) + ';' + cnn.identifier + ';' + str(results[0]) + ';' + str(results[1]) + '\n')
