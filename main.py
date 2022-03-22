import numpy as np
import random

from classifier import *
from util import *

inputs, labels = read_data('2d.trn.dat')
inputs -= inputs.mean(axis=1, keepdims=True)
inputs /= inputs.std(axis=1, keepdims=True)

test_inputs, test_labels = read_data('2d.tst.dat')
test_inputs -= test_inputs.mean(axis=1, keepdims=True)
test_inputs /= test_inputs.std(axis=1, keepdims=True)

(dim, count) = inputs.shape

indices = np.arange(count)
random.shuffle(indices)
split = int(0.8*count)

train_indices = indices[:split] 
valid_indices  = indices[split:]

train_inputs = inputs[:, train_indices]
train_labels = labels[train_indices]

valid_inputs = inputs[:, valid_indices]
valid_labels = labels[valid_indices]

print('Data shape after loading and spliting:')
print('train_inputs: {} train_labels: {}'.format(train_inputs.shape, train_labels.shape))
print('valid_inputs: {} valid_labels: {}'.format(valid_inputs.shape, valid_labels.shape))
print('test_inputs: {} test_labels: {}'.format(test_inputs.shape, test_labels.shape))

plot_dots(train_inputs, train_labels, None, test_inputs, test_labels, None)

model = MLPClassifier(dim_in=dim, dim_hid=20, n_classes=np.max(labels)+1)

trainCEs, trainREs = model.train(train_inputs, train_labels, alpha=0.1, eps=200, live_plot=False, live_plot_interval=25)

testCE, testRE = model.test(test_inputs, test_labels)
print('Final testing error: CE = {:6.2%}, RE = {:.5f}'.format(testCE, testRE))

_, train_predicted = model.predict(train_inputs)
_, test_predicted  = model.predict(test_inputs)

plot_dots(train_inputs, train_labels, train_predicted, test_inputs, test_labels, test_predicted, block=False)
plot_dots(None, None, None, test_inputs, test_labels, test_predicted, title='Test data only', block=False)
plot_both_errors(trainCEs, trainREs, testCE, testRE, block=False)