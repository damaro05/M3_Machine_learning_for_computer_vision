import os
import numpy as np
import shutil

from skimage.io import imread

dataset_dir = '/share/datasets/MIT_split'
destination_dir = '/share/datasets/MIT_split'
validation_size = .3
rand_seed = 2018  # None

if not os.path.exists(dataset_dir):
    print(dataset_dir + ' doesn\'t exist')
    exit(-1)

np.random.seed(rand_seed)
mean = []
precomp_mean = [109.07621812, 115.45609435, 114.70990406]
std = []
for class_folder in [f for f in os.listdir(os.path.join(dataset_dir, 'train')) if not f.startswith('.')]:
    print('Reading class ' + class_folder)

    # Read all image filenames and sort them alphabetically
    images = sorted([f for f in os.listdir(os.path.join(dataset_dir, 'train', class_folder)) if not f.startswith('.')])

    for im in images:
        X = imread(os.path.join(dataset_dir, 'train', class_folder,im))
        curmean = np.mean(X,axis=(0,1))
        mean.append(curmean)
        if not precomp_mean==None:
            X = X.astype('float64')
            X[:,:,0] -= precomp_mean[0]
            X[:,:,1] -= precomp_mean[1]
            X[:,:,2] -= precomp_mean[2]
            curstd = np.std(X,axis=(0,1))
            std.append(curstd)

print('Done!')
# We are able to compute it like this because all images have the same size
mean = np.mean(mean,axis=0)
if not precomp_mean == None:
    std = np.mean(std,axis=0)
print('Mean: '+str(mean))
if not precomp_mean == None:
    print('Std: '+str(std))
# Mean: [109.07621812 115.45609435 114.70990406]
# Std:  [56.91689916 55.4694083  59.14847488]
