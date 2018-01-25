import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import os

def plot_history(history, dirpath, identifier):

    # summarize history for accuracy
    plt.plot(history.history['acc'])
    if 'val_acc' in history.history:
        plt.plot(history.history['val_acc'])
    plt.title('model accuracy')
    plt.ylabel('accuracy')
    plt.xlabel('epoch')
    if 'val_acc' in history.history:
        plt.legend(['train', 'validation'], loc='upper left')
    plt.savefig(os.path.join(dirpath, identifier + '_accuracy.jpg'))
    plt.close()

    # summarize history for loss
    plt.plot(history.history['loss'])
    if 'val_loss' in history.history:
        plt.plot(history.history['val_loss'])
    plt.title('model loss')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    if 'val_loss' in history.history:
        plt.legend(['train', 'validation'], loc='upper left')
    plt.savefig(os.path.join(dirpath, identifier + '_loss.jpg'))