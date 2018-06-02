from __future__ import print_function
import time
import numpy as np
import matplotlib.pyplot as plt
from cs231n.classifiers.fc_net import *
from cs231n.data_utils import get_CIFAR10_data
from cs231n.gradient_check import eval_numerical_gradient, eval_numerical_gradient_array
from cs231n.solver import Solver

plt.rcParams['figure.figsize'] = (10.0, 8.0)
plt.rcParams['image.interpolation'] = 'nearest'
plt.rcParams['image.cmap'] = 'gray'

def rel_error(x, y):
    return np.max(np.abs(x - y)/(np.maximum(1e-8, np.abs(x) + np.abs(y))))

data = get_CIFAR10_data()
for k,v in list(data.items()):
    print(('%s: ' % k, v.shape))

learning_rate = [5e-4]
weight_scale = [2e-2]
best_model = None
best_acc = 0
for wei in weight_scale:
    for lr in learning_rate:
        model = FullyConnectedNet([100,100,100,100], weight_scale=wei)

        solver = Solver(model, data, update_rule='rmsprop', 
                optim_config={'learning_rate':lr,},
                lr_decay=0.8, 
                num_epochs=12, 
                batch_size=100, 
                print_every=100)

        solver.train()
        scores = model.loss(data['X_test'])
        y_pred = np.argmax(scores, axis=1)
        acc = np.mean(y_pred == data['y_test'])
        print ('test acc: %f'% (acc))
        if acc > best_acc:
            best_model = solver

plt.subplot(2,1,1)
plt.title('Training loss')
plt.plot(best_model.loss_history, 'o')
plt.xlabel('Iteration')

plt.subplot(2,1,2)
plt.title('Accuracy')
plt.plot(best_model.train_acc_history, '-o', label='train')
plt.plot(best_model.val_acc_history, '-o', label='val')
plt.plot([0.5]*len(best_model.val_acc_history), 'k--')
plt.xlabel('Epoch')
plt.legend(loc='lower right')
plt.gcf().set_size_inches(15,12)
plt.show()
