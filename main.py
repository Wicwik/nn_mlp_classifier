import numpy as np

from classifier import *
from util import *

train_inputs, train_labels = read_data('2d.trn.dat')

test_inputs, test_labels = read_data('2d.tst.dat')

print('Data shape after loading and categorical conversion:')
print('train_inputs: {} train_labels: {}'.format(train_inputs.shape, train_labels.shape))
print('test_inputs: {} test_labels: {}'.format(test_inputs.shape, test_labels.shape))

plot_dots(train_inputs, train_labels, None, test_inputs, test_labels, None)

(dim, count) = train_inputs.shape

model = MLPClassifier(dim_in=dim, dim_hid=20, n_classes=np.max(train_labels)+1)

trainCEs, trainREs = model.train(train_inputs, train_labels, alpha=0.1, eps=500, live_plot=False, live_plot_interval=25)