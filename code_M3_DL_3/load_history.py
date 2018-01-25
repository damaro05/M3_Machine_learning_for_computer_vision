import cPickle

import os

class ExecutionInfo:
    def __init__(self,born_time,hash,dump_path='dump'):
        path = os.path.join(dump_path,str(born_time)+'_'+str(hash))
        hfile = [f for f in os.listdir(path) if '.pklz' in f][0]
        with open(os.path.join(path,hfile), 'rb') as f:
            (self.epoch,self.history,self.params,self.validation_data,self.model_config) = cPickle.load(f)

'''#with open('./dump0/histories/task_1_-3171259397243059701_32_20_history.pklz','rb') as f:
    history = cPickle.load(f)
print(history[1]['val_acc'][-1])
print(history[1]['val_loss'][-1])
history
'''