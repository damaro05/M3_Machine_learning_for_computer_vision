import cPickle
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.lines as mlines
import numpy as np
from load_history import ExecutionInfo
import os
cnnexecutions = pd.read_csv('architecture_alternatives/log.csv',index_col='born_time',names=['born_time','type','hash','val_loss','val_acc'],sep=';')

train_acc = []
train_loss = []
idx=[]
for born_time in cnnexecutions.index:
    hash = cnnexecutions.loc[born_time, 'hash']
    exinfo = ExecutionInfo(born_time,hash,dump_path='architecture_alternatives/dump')
    idx.append(born_time)
    train_acc.append(exinfo.history['acc'][-1])
    train_loss.append(exinfo.history['loss'][-1])
cnnexecutions['acc']= pd.Series(train_acc, index=idx)
cnnexecutions['loss']= pd.Series(train_acc, index=idx)

for type in np.unique(cnnexecutions['type']):
    plt.figure()

    boxplot_elements=plt.boxplot(cnnexecutions.loc[cnnexecutions['type'] == type, 'acc'].values)
    for element in ['boxes','whiskers','caps','fliers','means']:
        plt.setp(boxplot_elements[element], color='C0')
    plt.setp(boxplot_elements['medians'], color='C3')

    boxplot_elements=plt.boxplot(cnnexecutions.loc[cnnexecutions['type'] == type, 'val_acc'].values)
    for element in ['boxes','whiskers','caps','fliers','means']:
        plt.setp(boxplot_elements[element], color='C1')
    plt.setp(boxplot_elements['medians'], color='C2')
    train_line = mlines.Line2D([], [], color='C0', label='train')
    val_line = mlines.Line2D([], [], color='C1',label='validation')
    plt.legend(handles=[train_line,val_line], loc='upper right')
    plt.grid(True)
    plt.ylim(0, 1)
    plt.title('Type ' + str(type))
plt.show()


# Boxplots
def boxplots():
    for type in np.unique(cnnexecutions['type']):
        plt.figure()
        cnnexecutions[cnnexecutions['type']==type].boxplot(column='val_acc')
        plt.ylim(0,1)
        plt.title('Type '+str(type))
    plt.show()

# Function to modify the log.csv file in case there is an error in every line
def clean_log(infile,outfile):
    f=open(infile,'r')
    lines=f.readlines()
    f.close()
    f=open(outfile,'w')
    for line in lines:
        # Do something
        f.write(line[:-2]+'\n')
    f.close()