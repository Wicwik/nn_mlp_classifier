# Robert Belanec
# I tried to write my code as self-explanatory as possible

import numpy as np
import random

from classifier import *
from util import *


CONFIG_FILE = 'hyperparameters.json'

# Load and normalize train data
inputs, labels = read_data('2d.trn.dat')
(dim, count) = inputs.shape

inputs -= inputs.mean(axis=1, keepdims=True)
inputs /= inputs.std(axis=1, keepdims=True)

# Load and normalize test data
test_inputs, test_labels = read_data('2d.tst.dat')
test_inputs -= test_inputs.mean(axis=1, keepdims=True)
test_inputs /= test_inputs.std(axis=1, keepdims=True)

# Get some idea abot the data, by ploting it
plot_dots(inputs, labels, None, test_inputs, test_labels, None, filename='all_data.png')

configs = get_hyperparameter_configurations(CONFIG_FILE)
models = []
best_model_index = None
best_model_validCE = np.inf

for idx,conf in enumerate(configs):
	# Split train data to train and validation (20% of train data is validation)
	indices = np.arange(count)
	random.shuffle(indices)
	split = int(0.8*count)

	train_indices = indices[:split] 
	valid_indices  = indices[split:]

	train_inputs = inputs[:, train_indices]
	train_labels = labels[train_indices]

	valid_inputs = inputs[:, valid_indices]
	valid_labels = labels[valid_indices]

	# Model creation
	model = MLPClassifier(dim_in=dim, dim_hid=conf['dim_hid'], w_init=conf['w_init'], optimizer=conf['optimizer'], f_hid=conf['f_hid'], f_out=conf['f_out'], n_classes=np.max(labels)+1)

	decay = None
	if conf['lr_schedule']:
		decay = conf['alpha']/conf['eps']

	# Training
	print('\nTraining of {}. model:'.format(idx+1))
	trainCEs, trainREs = model.train(train_inputs, train_labels, alpha=conf['alpha'], eps=conf['eps'], batchsize=conf['batchsize'], decay=decay)

	# Validation of the hyperparameters
	validCE, validRE = model.test(valid_inputs, valid_labels)

	print('{}. model valid error: CE = {:6.2%}, RE = {:.5f}'.format(idx+1, validCE, validRE))
	with open('training_results.txt', 'a') as f:
		f.write('{}. model valid error: CE = {:6.2%}, RE = {:.5f}\n'.format(idx+1, validCE, validRE))

	plot_both_errors(trainCEs, trainREs, validCE, validRE, block=False, filename='model_{}_errors.png'.format(idx+1))


	models.append((model, {'train_err': (trainCEs, trainREs), 'valid_err': (validCE, validRE)}))

	if (validCE < best_model_validCE):
		best_model_index = idx
		best_model_validCE = validCE


print('{}. model had the best validation score of CE = {:6.2%}, RE = {:.5f}.'.format(best_model_index+1, models[best_model_index][1]['valid_err'][0], models[best_model_index][1]['valid_err'][1]))
model = models[best_model_index][0]

# Final testing of the best model
testCE, testRE = model.test(test_inputs, test_labels)

print('Final testing error: CE = {:6.2%}, RE = {:.5f}'.format(testCE, testRE))
with open('test_results.txt', 'a') as f:
	f.write('Final testing error: CE = {:6.2%}, RE = {:.5f}'.format(testCE, testRE))

_, train_predicted = model.predict(train_inputs)
_, test_predicted  = model.predict(test_inputs)

plot_confusion_matrix(test_labels, test_predicted)

plot_dots(train_inputs, train_labels, train_predicted, test_inputs, test_labels, test_predicted, block=False, filename='all_data_predicted.png')
plot_dots(None, None, None, test_inputs, test_labels, test_predicted, title='Test data only', block=False, filename='test_data_predicted.png')
