# This file was adopted from NN seminars and changed (to fit project purposes) by Robert Belanec
# Neural Networks (2-AIN-132/15), FMFI UK BA
# (c) Tomas Kuzma, Juraj Holas, Peter Gergel, Endre Hamerlik, Stefan Pocos, Iveta Bečková 2017-2022

import numpy as np

from mlp import *
from util import *


class MLPClassifier(MLP):

    def __init__(self, dim_in, dim_hid, n_classes, w_init='gauss', optimizer='sgd', f_hid='relu', f_out='sigmoid'):
        self.n_classes = n_classes
        self.optimizer = optimizer
        self.f_hid_selected = f_hid
        self.f_out_selected = f_out

        super().__init__(dim_in, dim_hid, dim_out=n_classes, w_init=w_init, optimizer=optimizer)


    # Activation functions & derivations

    def error(self, targets, outputs):
        '''
        Cost / loss / error function
        '''
        return np.sum((targets - outputs)**2)

    # @override
    def f_hid(self, x):
        if self.f_hid_selected == 'relu':
            return x * (x > 0)
        elif self.f_hid_selected == 'sigmoid':
            return 1 / (1 + np.exp(-x))
        elif self.f_hid_selected == 'tanh':
            return np.tanh(x)

    # @override
    def df_hid(self, x):
        if self.f_hid_selected == 'relu':
            return 1. * (x > 0)
        elif self.f_hid_selected == 'sigmoid':
            return self.f_hid(x)*(1 - self.f_hid(x)) 
        elif self.f_hid_selected == 'tanh':
            return 1 - np.tanh(x)**2

    # @override
    def f_out(self, x):
        if self.f_out_selected == 'sigmoid':
            return 1 / (1 + np.exp(-x))
        elif self.f_out_selected == 'linear':
            return x

    # @override
    def df_out(self, x):
        if self.f_out_selected == 'sigmoid':
            return self.f_out(x)*(1 - self.f_out(x)) 
        elif self.f_out_selected == 'linear':
            return 1


    def minibatches(self, inputs, targets, batchsize, shuffle=False):
        '''
        Creates minibatches and lazy loads them 
        inputs: input data
        targets: target labels
        batchsize: size of mini batch
        shuffle: shuffle data
        '''
        assert inputs.shape[1] == targets.shape[1]

        if shuffle:
            indices = np.arange(inputs.shape[1])

        for start_idx in range(0, inputs.shape[1] - batchsize + 1, batchsize):
            if shuffle:
                excerpt = indices[start_idx:start_idx + batchsize]
            else:
                excerpt = slice(start_idx, start_idx + batchsize)
            yield inputs[:, excerpt], targets[:, excerpt], excerpt



    def predict(self, inputs):
        '''
        Prediction = forward pass
        '''
        # If self.forward() can process only one input at a time
        outputs = np.stack([self.forward(x)[-1] for x in inputs.T]).T
        # # If self.forward() can take a whole batch
        # *_, outputs = self.forward(inputs)
        return outputs, onehot_decode(outputs)


    def test(self, inputs, labels):
        '''
        Test model: forward pass on given inputs, and compute errors
        '''
        targets = onehot_encode(labels, self.n_classes)          
        outputs, predicted = self.predict(inputs)

        CE = np.sum((predicted != labels))/labels.shape[0]       
        RE = np.mean((targets-outputs)**2)

        return CE, RE


    def train(self, inputs, labels, alpha=0.1, eps=100, batchsize=100, decay=None, beta1=0.9, beta2=0.999, epsilon=10e-08, live_plot=False, live_plot_interval=10):
        '''
        Training of the classifier
        inputs: matrix of input vectors (each column is one input vector)
        labels: vector of labels (each item is one class label)
        alpha: learning rate
        eps: number of episodes
        live_plot: plot errors and data during training
        live_plot_interval: refresh live plot every N episodes
        '''
        (_, count) = inputs.shape
        targets = onehot_encode(labels, self.n_classes)

        if live_plot:
            interactive_on()

        CEs = []
        REs = []

        for ep in range(eps):
            CE = 0
            RE = 0

            # no cycles and minibatches -> very fast learning
            for batch in self.minibatches(inputs, targets, batchsize):
                x, d, idx = batch

                a, h, b, y = self.forward(x)

                if self.optimizer == 'sgd':
                    dW_hid, dW_out = self.backward(x, a, h, b, y, d)

                    self.W_hid += alpha * dW_hid
                    self.W_out += alpha * dW_out

                elif self.optimizer == 'adam':
                    m_dW_hid_corr, m_dW_out_corr, v_dW_hid_corr, v_dW_out_corr = self.backward_adam(x, a, h, b, y, d, ep+1, beta1, beta2, epsilon)

                    # calculate the weight difference from Adam moments
                    self.W_hid += alpha*(m_dW_hid_corr/(np.sqrt(v_dW_hid_corr)+epsilon))
                    self.W_out += alpha*(m_dW_out_corr/(np.sqrt(v_dW_out_corr)+epsilon))

                CE += sum(labels[idx] != onehot_decode(y))
                RE += self.error(d, y)

            CE /= count
            RE /= count
            CEs.append(CE)
            REs.append(RE)

            if decay is not None:
                alpha = alpha * 1/(1 + decay * ep)

            if (ep+1) % 5 == 0: print('Epoch {:3d}/{}, CE = {:6.2%}, RE = {:.5f}'.format(ep+1, eps, CE, RE))

            if live_plot and ((ep+1) % live_plot_interval == 0):
                _, predicted = self.predict(inputs)
                plot_dots(inputs, labels, predicted, block=False)
                plot_both_errors(CEs, REs, block=False)
                plot_areas(self, inputs, block=False)
                redraw()

        if live_plot:
            interactive_off()


        return CEs, REs