import itertools
import numpy as np
import matplotlib.pyplot as plt

from sklearn.model_selection import KFold
from sklearn.metrics import confusion_matrix
from sklearn.metrics import precision_recall_fscore_support
from sklearn.metrics import average_precision_score
from sklearn import metrics
from itertools import cycle


def confusionMatrix(Gtruth, predicted, graph=False, normalization=False):
    """ This fucntion calculates the confusion matrix, normalization
        can be applied by setting 'normalization'=True and plot the matrix 
        by setting 'graph'=True """
    classes = ['mountain', 'inside_city', 'Opencountry', 'coast', 'street', 'forest', 'tallbuilding', 'highway']
    cm = confusion_matrix(Gtruth, predicted, classes)
    if graph is False:
        return cm

    plot_confusion_matrix(cm, classes, normalization)

    return cm
    

def plot_confusion_matrix(cm, classes, normalize=False, title='Confusion matrix', cmap=plt.cm.Blues):
    """ This function prints and plots the confusion matrix.
        Normalization can be applied by setting 'normalize'=True. """
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
        title='Normalized confusion matrix'
    else:
        print('Confusion matrix, without normalization')
    #print(cm)
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], fmt),
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.show()


def plot_roc_curve(Gtruth, predicted, classifier):
    """ This function plots the ROC curve of each class.
        A classifier... """
    classes = ['mountain', 'inside_city', 'Opencountry', 'coast', 'street', 'forest', 'tallbuilding', 'highway']
    colors = cycle(['aqua', 'darkorange', 'cornflowerblue'])
    # Binarize the output
    #Gtruth = label_binarize(Gtruth, classes=[0, 1, 2, 3, 4, 5, 6, 7])
    #n_classes = Gtruth.shape[1]

    probas_ = classifier.predict_proba(predicted)

    for i in range(len(classes)):
        fpr, tpr, thresholds = metrics.roc_curve(Gtruth, probas_[:,1], classes[i])

        roc_auc = metrics.auc(fpr, tpr)
        plt.title('Receiver Operating Characteristic')
        plt.plot(fpr, tpr, label = classes[i] % roc_auc)
        plt.legend(loc = 'lower right')
        plt.plot([0, 1], [0, 1],'r--')
        plt.xlim([0, 1])
        plt.ylim([0, 1])
        plt.ylabel('True Positive Rate')
        plt.xlabel('False Positive Rate')
    plt.show()


def precision_recall_fscore(Gtruth, predicted, averageP=False):
    """ This function calculates the precision, recall, f-score and support.
        The average precision can be calculated by setting 'averageP'=True """
    precision, recall, fscore, support = precision_recall_fscore_support(Gtruth, predicted)
    average_precision = np.mean(precision)
    if averageP is not False:
        return average_precision, recall, fscore, support
    return precision, recall, fscore, support

    
def plot_precision_recall_curve():
    """ No tested yet """
    colors = cycle(['navy', 'turquoise', 'darkorange', 'cornflowerblue', 'teal'])

    plt.figure(figsize=(7, 8))
    f_scores = np.linspace(0.2, 0.8, num=4)
    lines = []
    labels = []
    for f_score in f_scores:
        x = np.linspace(0.01, 1)
        y = f_score * x / (2 * x - f_score)
        l, = plt.plot(x[y >= 0], y[y >= 0], color='gray', alpha=0.2)
        plt.annotate('f1={0:0.1f}'.format(f_score), xy=(0.9, y[45] + 0.02))

    lines.append(l)
    labels.append('iso-f1 curves')
    l, = plt.plot(recall["micro"], precision["micro"], color='gold', lw=2)
    lines.append(l)
    labels.append('micro-average Precision-recall (area = {0:0.2f})'
                  ''.format(average_precision["micro"]))

    for i, color in zip(range(n_classes), colors):
        l, = plt.plot(recall[i], precision[i], color=color, lw=2)
        lines.append(l)
        labels.append('Precision-recall for class {0} (area = {1:0.2f})'
                      ''.format(i, average_precision[i]))

    fig = plt.gcf()
    fig.subplots_adjust(bottom=0.25)
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title('Extension of Precision-Recall curve to multi-class')
    plt.legend(lines, labels, loc=(0, -.38), prop=dict(size=14))
    plt.show()
