from __future__ import print_function
import random
import numpy as np
import matplotlib.pyplot as plt
from cs231n.data_utils import load_CIFAR10

plt.rcParams['image.interpolation'] = 'nearest'
plt.rcParams['image.cmap'] = 'gray'

def get_CIFAR10_data(num_training=49000, num_validation=1000, num_test=1000):
    cifar10_dir = 'cs231n/datasets/cifar-10-batches-py'
    X_train, y_train, X_test, y_test = load_CIFAR10(cifar10_dir)
    
    mask = list(range(num_training, num_training+num_validation))
    X_val = X_train[mask]
    y_val = y_train[mask]

    mask = list(range(num_training))
    X_train = X_train[mask]
    y_train = y_train[mask]

    mask = list(range(num_test))
    X_test = X_test[mask]
    y_test = y_test[mask]

    return X_train, y_train, X_val, y_val, X_test, y_test

try:
    del X_train, y_train
    del X_val, y_val
    del X_test, y_test
    print('Clear previously loaded data.')
except:
    pass

X_train, y_train, X_val, y_val, X_test, y_test = get_CIFAR10_data()

from cs231n.features import *

num_color_bins = 10
feature_fns = [hog_feature, lambda img: color_histogram_hsv(img, nbin=num_color_bins)]
X_train_feats = extract_features(X_train, feature_fns, verbose=True)
X_val_feats = extract_features(X_val, feature_fns)
X_test_feats = extract_features(X_test, feature_fns)

mean_feat = np.mean(X_train_feats, axis=0, keepdims=True)
X_train_feats -= mean_feat
X_val_feats -= mean_feat
X_test_feats -= mean_feat

std_feat = np.std(X_train_feats, axis=0, keepdims=True)
X_train_feats /= std_feat
X_val_feats /= std_feat
X_test_feats /= std_feat

X_train_feats = np.hstack([X_train_feats, np.ones((X_train_feats.shape[0], 1))])
X_val_feats = np.hstack([X_val_feats, np.ones((X_val_feats.shape[0], 1))])
X_test_feats = np.hstack([X_test_feats, np.ones((X_test_feats.shape[0], 1))])

from cs231n.classifiers.linear_classifier import LinearSVM

learning_rates = [1e-9, 1e-8, 1e-7]
regularization_strengths = [5e4, 5e5, 5e6]

results = {}
best_val = -1
best_svm = None

for lr in learning_rates:
    for reg in regularization_strengths:
        svm = LinearSVM()
        svm.train(X_train_feats, y_train, learning_rate=lr, reg=reg, num_iters=6000)
      
        train_pred = svm.predict(X_train_feats)
        val_pred = svm.predict(X_val_feats)
        train_accuracy = np.mean(train_pred == y_train)
        val_accuracy = np.mean(val_pred == y_val)
       
        if val_accuracy > best_val:
            best_val = val_accuracy
            best_svm = svm

        results[(lr, reg)] = (train_accuracy, val_accuracy)

for lr, reg in sorted(results):
    train_accuracy, val_accuracy = results[(lr, reg)]
    print('lr %e reg %e train accuracy: %f val accuracy: %f' % (lr, reg, train_accuracy, val_accuracy))

print('best validation accuracy achieved during cross-validation: %f' % best_val)


y_test_pred = best_svm.predict(X_test_feats)
test_accuracy = np.mean(y_test_pred == y_test)
print(test_accuracy)

examples_per_class = 8
classes = ['plane','car','bird','cat','deer','dog','frog','horse','ship','truck']
for cls, cls_name in enumerate(classes):
    idxs = np.where((y_test != cls) & (y_test_pred == cls))[0]
    idxs = np.random.choice(idxs, examples_per_class, replace=False)
    for i, idx in enumerate(idxs):
        plt.subplot(examples_per_class, len(classes), i*len(classes)+cls+1)
        plt.imshow(X_test[idx].astype('uint8'))
        plt.axis('off')
        if i == 0:
            plt.title(cls_name)
plt.show()

print(X_train_feats.shape)
X_train_feats = X_train_feats[:,:-1]
X_val_feats = X_val_feats[:,:-1]
X_test_feats = X_test_feats[:,:-1]
print(X_train_feats.shape)

from cs231n.classifiers.neural_net import TwoLayerNet

input_dim = X_train_feats.shape[1]
hidden_dim = 500
num_classes = 10

best_net = None
best_val = -1
learning_rates = [1e-9, 1e-8, 1e-7]
regularization_strengths = [5e4, 5e5, 5e6]

for lr in learning_rates:
    for reg in regularization_strengths:
        net = TwoLayerNet(input_dim, hidden_dim, num_classes)
        net.train(X_train_feats, y_train, X_val_feats, y_val,
                learning_rate=lr, learning_rate_decay=0.95,
                reg=reg, num_iters=5000, batch_size=200, verbose=False)
       
        train_accuracy = (net.predict(X_train_feats) == y_train).mean()
        val_accuracy = (net.predict(X_val_feats) == y_val).mean()
        print('lr %e reg %e train accuracy: %f val accuracy: %f' % (lr,reg,train_accuracy,val_accuracy))
        if val_accuracy > best_val:
            best_val = val_accuracy
            best_net = net

print('best validation accuracy achieved during cross-validation: %f' % best_val)

test_accuracy = (best_net.predict(X_test_feats) == y_test).mean()
print(test_accuracy)
