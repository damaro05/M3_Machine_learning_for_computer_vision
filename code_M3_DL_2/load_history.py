import cPickle
with open('./dump0/histories/task_1_-3171259397243059701_32_20_history.pklz','rb') as f:
    history = cPickle.load(f)
print(history[1]['val_acc'][-1])
print(history[1]['val_loss'][-1])
history