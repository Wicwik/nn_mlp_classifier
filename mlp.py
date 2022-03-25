# This file was adopted from NN seminars and changed (to fit project purposes) by Robert Belanec
# Neural Networks (2-AIN-132/15), FMFI UK BA
# (c) Tomas Kuzma, Juraj Holas, Peter Gergel, Endre Hamerlik, Stefan Pocos, Iveta Bečková 2017-2022

import numpy as np

from util import *


class MLP():
    '''
    Multi-Layer Perceptron (abstract base class)
    '''

    def __init__(self, dim_in, dim_hid, dim_out, w_init='gauss', optimizer='sgd'):
        '''
        Initialize model, set initial weights
        '''
        self.dim_in  = dim_in
        self.dim_hid = dim_hid
        self.dim_out = dim_out

        if w_init == 'gauss':
            self.W_hid = np.random.randn(dim_hid, dim_in + 1)
            self.W_out = np.random.randn(dim_out, dim_hid + 1)

        elif w_init == 'uniform':
            self.W_hid = np.random.rand(dim_hid, dim_in + 1)
            self.W_out = np.random.rand(dim_out, dim_hid + 1)


        if optimizer == 'adam':
            self.m_dW_hid, self.m_dW_out, self.v_dW_hid, self.v_dW_out = 0, 0, 0, 0

    # Activation functions & derivations
    # (not implemented, to be overriden in derived classes)
    def f_hid(self, x):
        raise NotImplementedError

    def df_hid(self, x):
        raise NotImplementedError

    def f_out(self, x):
        raise NotImplementedError

    def df_out(self, x):
        raise NotImplementedError


    # Back-propagation

    def forward(self, x):
        '''
        Forward pass - compute output of network
        x: single input vector (without bias, size=dim_in)
        '''
        a = self.W_hid @ add_bias(x)
        h = self.f_hid(a)
        b = self.W_out @ add_bias(h)
        y = self.f_out(b)

        return a, h, b, y


    def backward(self, x, a, h, b, y, d):
        '''
        Backprop pass - compute dW for given input and activations
        x: single input vector (without bias, size=dim_in)
        a: net vector on hidden layer (size=dim_hid)
        h: activation of hidden layer (without bias, size=dim_hid)
        b: net vector on output layer (size=dim_out)
        y: output vector of network (size=dim_out)
        d: single target vector (size=dim_out)
        '''

        g_out = (d - y) * self.df_out(b)


        g_hid = self.W_out[:, :self.dim_hid].T@g_out * self.df_hid(a)

        dW_out = np.dot(add_bias(h), g_out.T).T
        dW_hid = np.dot(add_bias(x), g_hid.T).T

        return dW_hid, dW_out

    # not working yet :(
    def backward_adam(self, x, a, h, b, y, d, ep, beta1, beta2, epsilon):
        g_out = (d - y) * self.df_out(b)
        g_hid = self.W_out[:, :self.dim_hid].T@g_out * self.df_hid(a)
        
        dW_out = np.dot(add_bias(h), g_out.T).T
        dW_hid = np.dot(add_bias(x), g_hid.T).T        

        self.m_dW_out = beta1*self.m_dW_out + (1-beta1)*dW_out
        self.m_dW_hid = beta1*self.m_dW_hid + (1-beta1)*dW_hid

        self.v_dW_out = beta2*self.v_dW_out + (1-beta2)*(dW_out**2)
        self.v_dW_hid = beta2*self.v_dW_hid + (1-beta2)*(dW_hid**2)

        m_dW_out_corr = self.m_dW_out/(1-beta1**ep)
        m_dW_hid_corr = self.m_dW_hid/(1-beta1**ep)
        
        v_dW_out_corr = self.v_dW_out/(1-beta2**ep)
        v_dW_hid_corr = self.v_dW_hid/(1-beta2**ep)
        
        return m_dW_hid_corr, m_dW_out_corr, v_dW_hid_corr, v_dW_out_corr