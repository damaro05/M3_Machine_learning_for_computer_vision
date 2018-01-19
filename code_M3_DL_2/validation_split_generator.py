import os
import numpy as np
import shutil

dataset_dir = '/share/datasets/MIT_split'
destination_dir = '/share/datasets/MIT_split'
validation_size = .3
rand_seed = 2018  # None

if not os.path.exists(dataset_dir):
    print(dataset_dir + ' doesn\'t exist')
    exit(-1)

np.random.seed(rand_seed)
for class_folder in [f for f in os.listdir(os.path.join(dataset_dir, 'train')) if not f.startswith('.')]:
    print('Spliting class ' + class_folder)

    # Read all image filenames and sort them alphabetically
    images = sorted([f for f in os.listdir(os.path.join(dataset_dir, 'train', class_folder)) if not f.startswith('.')])

    # Permute the list
    images = np.random.permutation(images)

    # Split the list
    validation_images = images[:int(len(images) * validation_size)]
    train_images = images[int(len(images) * validation_size):]

    # Copy the files
    if not os.path.exists(os.path.join(destination_dir, 'train_split', class_folder)):
        os.makedirs(os.path.join(destination_dir, 'train_split', class_folder))
    if not os.path.exists(os.path.join(destination_dir, 'validation_split', class_folder)):
        os.makedirs(os.path.join(destination_dir, 'validation_split', class_folder))

    for im in train_images:
        shutil.copy(os.path.join(dataset_dir, 'train', class_folder, im),
                    os.path.join(destination_dir, 'train_split', class_folder))
    for im in validation_images:
        shutil.copy(os.path.join(dataset_dir, 'train', class_folder, im),
                    os.path.join(destination_dir, 'validation_split', class_folder))

    print('Done!')
