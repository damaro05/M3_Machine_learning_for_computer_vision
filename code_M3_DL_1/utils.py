from __future__ import print_function
import matplotlib

matplotlib.use('Agg')
import matplotlib.pyplot as plt
import os, sys
import numpy as np
from sklearn.feature_extraction import image
from PIL import Image



def softmax(x):
    """Compute softmax values for each sets of scores in x."""
    e_x = np.exp(x - np.max(x))
    return e_x / e_x.sum(axis=0)


class Color:
    GRAY = 30
    RED = 31
    GREEN = 32
    YELLOW = 33
    BLUE = 34
    MAGENTA = 35
    CYAN = 36
    WHITE = 37
    CRIMSON = 38


def colorize(num, string, bold=False, highlight=False):
    assert isinstance(num, int)
    attr = []
    if highlight: num += 10
    attr.append(str(num))
    if bold: attr.append('1')
    return '\x1b[%sm%s\x1b[0m' % (';'.join(attr), string)


def colorprint(colorcode, text, o=sys.stdout, bold=False):
    o.write(colorize(colorcode, text, bold=bold))


def generate_image_patches_db(in_directory, out_directory, patch_size=64,max_patches=15,):
    if not os.path.exists(out_directory):
        os.makedirs(out_directory)

    total = 2688
    count = 0
    for split_dir in os.listdir(in_directory):
        if not os.path.exists(os.path.join(out_directory, split_dir)):
            os.makedirs(os.path.join(out_directory, split_dir))

        for class_dir in os.listdir(os.path.join(in_directory, split_dir)):
            print('Processed images: ' + str(count) + ' / ' + str(total))
            if not os.path.exists(os.path.join(out_directory, split_dir, class_dir)):
                os.makedirs(os.path.join(out_directory, split_dir, class_dir))

            for imname in os.listdir(os.path.join(in_directory, split_dir, class_dir)):
                count += 1
                #print('Processed images: ' + str(count) + ' / ' + str(total))
                im = Image.open(os.path.join(in_directory, split_dir, class_dir, imname))
                patches = image.extract_patches_2d(np.array(im), (patch_size, patch_size), max_patches=max_patches)
                for i, patch in enumerate(patches):
                    patch = Image.fromarray(patch)
                    patch.save(
                        os.path.join(out_directory, split_dir, class_dir, imname.split(',')[0] + '_' + str(i) + '.jpg'))
    print('Processed images: ' + str(count) + ' / ' + str(total))


def plot_history(history, model_identifier, metric='acc', plot_validation=False, path='histories'):
    plt.plot(history.history[metric])
    if plot_validation:
        plt.plot(history.history['val_' + metric])
    plt.title('model ' + metric)
    plt.ylabel(metric)
    plt.xlabel('epoch')
    if plot_validation:
        legend = ['train', 'validation']
    else:
        legend = ['train']
    plt.legend(legend, loc='upper left')
    plt.savefig(path+'/'+model_identifier + '_' + metric + '.png')
    plt.close()
