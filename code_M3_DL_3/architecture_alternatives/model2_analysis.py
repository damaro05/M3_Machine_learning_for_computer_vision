import cPickle
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.lines as mlines
import numpy as np
from load_history import ExecutionInfo
import os
cnnexecutions = pd.read_csv('log2.csv',index_col='born_time',names=['born_time','type','hash','val_loss','val_acc'],sep=';')
cnnexecutions=cnnexecutions[cnnexecutions['type']==2]
train_acc = []
train_loss = []
idx=[]
conv1_filters = []
conv2_filters = []
conv1_kernel = []
conv2_kernel = []
conv1_strides = []
conv2_strides = []
maxpool1_size = []
maxpool2_size = []
fc1_units = []
fc2_units = []
for born_time in cnnexecutions.index:
    hash = cnnexecutions.loc[born_time, 'hash']
    exinfo = ExecutionInfo(born_time,hash)
    idx.append(born_time)
    train_acc.append(exinfo.history['acc'][-1])
    train_loss.append(exinfo.history['loss'][-1])
    conv1_filters.append(exinfo.model_config[0]['config']['filters'])
    conv1_kernel.append(exinfo.model_config[0]['config']['kernel_size'][0])
    conv1_strides.append(exinfo.model_config[0]['config']['strides'][0])
    conv2_filters.append(exinfo.model_config[2]['config']['filters'])
    conv2_kernel.append(exinfo.model_config[2]['config']['kernel_size'][0])
    conv2_strides.append(exinfo.model_config[2]['config']['strides'][0])
    maxpool1_size.append(exinfo.model_config[1]['config']['pool_size'][0])
    maxpool2_size.append(exinfo.model_config[3]['config']['pool_size'][0])
    fc1_units.append(exinfo.model_config[5]['config']['units'])
    fc2_units.append(exinfo.model_config[6]['config']['units'])
cnnexecutions['acc']= pd.Series(train_acc, index=idx)
cnnexecutions['loss']= pd.Series(train_loss, index=idx)
cnnexecutions['conv1_filters']= pd.Series(conv1_filters, index=idx)
cnnexecutions['conv1_kernel']= pd.Series(conv1_kernel, index=idx)
cnnexecutions['conv1_strides']= pd.Series(conv1_strides, index=idx)
cnnexecutions['conv2_filters']= pd.Series(conv2_filters, index=idx)
cnnexecutions['conv2_kernel']= pd.Series(conv2_kernel, index=idx)
cnnexecutions['conv2_strides']= pd.Series(conv2_strides, index=idx)
cnnexecutions['maxpool1_size']= pd.Series(maxpool1_size, index=idx)
cnnexecutions['maxpool2_size']= pd.Series(maxpool2_size, index=idx)
cnnexecutions['fc1_units']= pd.Series(fc1_units, index=idx)
cnnexecutions['fc2_units']= pd.Series(fc2_units, index=idx)
cnnexecutions['bests']= cnnexecutions['val_acc']>np.percentile(cnnexecutions['val_acc'],85)


cnnexecutions.corr()
