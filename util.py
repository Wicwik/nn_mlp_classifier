# This file was adopted from NN seminars and changed (to fit project purposes) by Robert Belanec
# Neural Networks (2-AIN-132/15), FMFI UK BA
# (c) Tomas Kuzma, Juraj Holas, Peter Gergel, Endre Hamerlik, Stefan Pocos, Iveta Bečková 2017-2022

import matplotlib
matplotlib.use('TkAgg') # fixme if plotting doesn`t work (try 'Qt5Agg' or 'Qt4Agg')
import matplotlib.pyplot as plt
# for 3D visualization
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401 unused import
from matplotlib import cm
from sklearn.metrics import confusion_matrix

import numpy as np
import atexit
import os
import time
import functools
import json
import string


## Utilities

def onehot_decode(X):
    return np.argmax(X, axis=0)


def onehot_encode(L, c):
    if isinstance(L, int):
        L = [L]
    n = len(L)
    out = np.zeros((c, n))
    out[L, range(n)] = 1
    return np.squeeze(out)


def vector(array, row_vector=False):
    '''
    Construts a column vector (i.e. matrix of shape (n,1)) from given array/numpy.ndarray, or row
    vector (shape (1,n)) if row_vector = True.
    '''
    v = np.array(array)
    if np.squeeze(v).ndim > 1:
        raise ValueError('Cannot construct vector from array of shape {}!'.format(v.shape))
    return v.reshape((1, -1) if row_vector else (-1, 1))


def add_bias(X):
    '''
    Add bias term to vector, or to every (column) vector in a matrix.
    '''
    if X.ndim == 1:
        return np.concatenate((X, [1]))
    else:
        pad = np.ones((1, X.shape[1]))
        return np.concatenate((X, pad), axis=0)


def timeit(func):
    '''
    Profiling function to measure time it takes to finish function.
    Args:
        func(*function): Function to meassure
    Returns:
        (*function) New wrapped function with meassurment
    '''
    @functools.wraps(func)
    def newfunc(*args, **kwargs):
        start_time = time.time()
        out = func(*args, **kwargs)
        elapsed_time = time.time() - start_time
        print('Function [{}] finished in {:.3f} s'.format(func.__name__, elapsed_time))
        return out
    return newfunc



## Interactive drawing

def clear():
    plt.clf()


def interactive_on():
    plt.ion()
    plt.show(block=False)
    time.sleep(0.1)


def interactive_off():
    plt.ioff()
    plt.close()


def redraw():
    # plt.gcf().canvas.draw()   # fixme: uncomment if interactive drawing does not work
    plt.waitforbuttonpress(timeout=0.001)
    time.sleep(0.001)


def keypress(e):
    if e.key in {'q', 'escape'}:
        os._exit(0) # unclean exit, but exit() or sys.exit() won't work
    if e.key in {' ', 'enter'}:
        plt.close() # skip blocking figures


def use_keypress(fig=None):
    if fig is None:
        fig = plt.gcf()
    fig.canvas.mpl_connect('key_press_event', keypress)


## Non-blocking figures still block at end
def finish():
    plt.show(block=True) # block until all figures are closed


atexit.register(finish)



## Plotting
palette = ['#e41a1c','#377eb8','#4daf4a','#984ea3','#ff7f00','#ffff33','#a65628','#f781bf','#999999']


def limits(values, gap=0.05):
    x0 = np.min(values)
    x1 = np.max(values)
    xg = (x1 - x0) * gap
    return np.array((x0-xg, x1+xg))


def plot_errors(title, errors, test_error=None, block=True):
    plt.figure(1)
    use_keypress()
    plt.clf()
    plt.ylim(bottom=0)

    plt.plot(errors)

    if test_error:
        plt.plot([test_error]*len(errors))

    plt.tight_layout()
    plt.gcf().canvas.set_window_title(title)
    plt.show(block=block)


def plot_both_errors(trainCEs, trainREs, testCE=None, testRE=None, pad=None, figsize=(1918, 1025), block=True, filename=None):
    px = 1/plt.rcParams['figure.dpi']
    fig = plt.figure(2, figsize=tuple(i*px for i in figsize))
    use_keypress()
    plt.clf()

    if pad is None:
        pad = max(len(trainCEs), len(trainREs))
    else:
        trainCEs = np.concatentate((trainCEs, [None]*(pad-len(trainCEs))))
        trainREs = np.concatentate((trainREs, [None]*(pad-len(trainREs))))

    ax = plt.subplot(2,1,1)
    plt.ylim(bottom=0, top=100)
    plt.title('Classification error [%]')
    plt.plot(100*np.array(trainCEs), label='train set')

    if testCE is not None:
        plt.plot([100*testCE]*pad, label='valid set')
    plt.legend()

    plt.subplot(2,1,2)
    plt.ylim(bottom=0, top=1)
    plt.title('Model loss [MSE/sample]')
    plt.plot(trainREs, label='train set')

    if testRE is not None:
        plt.plot([testRE]*pad, label='test set')

    plt.tight_layout()
    plt.gcf().canvas.set_window_title('Error metrics')
    plt.legend()

    if filename is not None:
        plt.savefig(filename, dpi=fig.dpi)

    plt.show(block=block)


def plot_dots(inputs, labels=None, predicted=None, test_inputs=None, test_labels=None, test_predicted=None, s=60, i_x=0, i_y=1, title=None, figsize=(1918, 1025), block=True, filename=None):
    px = 1/plt.rcParams['figure.dpi']
    fig = plt.figure(title or 3, figsize=tuple(i*px for i in figsize))
    use_keypress()
    plt.clf()

    if inputs is not None:
        if labels is None:
            plt.gcf().canvas.set_window_title('Data distribution')
            plt.scatter(inputs[i_x,:], inputs[i_y,:], s=s, c=palette[-1], edgecolors=[0.4]*3, alpha=0.5, label='train data')

        elif predicted is None:
            plt.gcf().canvas.set_window_title('Class distribution')
            for i, c in enumerate(set(labels)):
                plt.scatter(inputs[i_x,labels==c], inputs[i_y,labels==c], s=s, c=palette[i], edgecolors=[0.4]*3, label='train cls {}'.format(c))

        else:
            plt.gcf().canvas.set_window_title('Predicted vs. actual')
            for i, c in enumerate(set(labels)):
                plt.scatter(inputs[i_x,labels==c], inputs[i_y,labels==c], s=2.0*s, c=palette[i], edgecolors=None, alpha=0.333, label='train cls {}'.format(c))

            for i, c in enumerate(set(labels)):
                plt.scatter(inputs[i_x,predicted==c], inputs[i_y,predicted==c], s=0.5*s, c=palette[i], edgecolors=None, label='predicted {}'.format(c))

        plt.xlim(limits(inputs[i_x,:]))
        plt.ylim(limits(inputs[i_y,:]))

    if test_inputs is not None:
        if test_labels is None:
            plt.scatter(test_inputs[i_x,:], test_inputs[i_y,:], marker='s', s=s, c=palette[-1], edgecolors=[0.4]*3, alpha=0.5, label='test data')

        elif test_predicted is None:
            for i, c in enumerate(set(test_labels)):
                plt.scatter(test_inputs[i_x,test_labels==c], test_inputs[i_y,test_labels==c], marker='s', s=s, c=palette[i], edgecolors=[0.4]*3, label='test cls {}'.format(c))

        else:
            for i, c in enumerate(set(test_labels)):
                plt.scatter(test_inputs[i_x,test_labels==c], test_inputs[i_y,test_labels==c], marker='s', s=2.0*s, c=palette[i], edgecolors=None, alpha=0.333, label='test cls {}'.format(c))

            for i, c in enumerate(set(test_labels)):
                plt.scatter(test_inputs[i_x,test_predicted==c], test_inputs[i_y,test_predicted==c], marker='s', s=0.5*s, c=palette[i], edgecolors=None, label='predicted {}'.format(c))

        if inputs is None:
            plt.xlim(limits(test_inputs[i_x,:]))
            plt.ylim(limits(test_inputs[i_y,:]))

    plt.legend()
    if title is not None:
        plt.gcf().canvas.set_window_title(title)
    plt.tight_layout()

    if filename is not None:
        plt.savefig(filename, dpi=fig.dpi)

    plt.show(block=block)


def plot_areas(model, inputs, labels=None, w=30, h=20, i_x=0, i_y=1, block=True):
    plt.figure(4)
    use_keypress()
    plt.clf()
    plt.gcf().canvas.set_window_title('Decision areas')

    dim = inputs.shape[0]
    data = np.zeros((dim, w*h))

    # # "proper":
    # X = np.linspace(*limits(inputs[i_x,:]), w)
    # Y = np.linspace(*limits(inputs[i_y,:]), h)
    # YY, XX = np.meshgrid(Y, X)
    #
    # for i in range(dim):
    #     data[i,:] = np.mean(inputs[i,:])
    # data[i_x,:] = XX.flat
    # data[i_y,:] = YY.flat

    X1 = np.linspace(*limits(inputs[0,:]), w)
    Y1 = np.linspace(*limits(inputs[1,:]), h)
    X2 = np.linspace(*limits(inputs[2,:]), w)
    Y2 = np.linspace(*limits(inputs[3,:]), h)
    YY1, XX1 = np.meshgrid(Y1, X1)
    YY2, XX2 = np.meshgrid(Y2, X2)
    data[0,:] = XX1.flat
    data[1,:] = YY1.flat
    data[2,:] = XX2.flat
    data[3,:] = YY2.flat

    outputs, *_ = model.predict(data)
    assert outputs.shape[0] == model.dim_out,\
           f'Outputs do not have correct shape, expected ({model.dim_out}, ?), got {outputs.shape}'
    outputs = outputs.reshape((-1,w,h))

    outputs -= np.min(outputs, axis=0, keepdims=True)
    outputs  = np.exp(1*outputs)
    outputs /= np.sum(outputs, axis=0, keepdims=True)

    plt.imshow(outputs.T)

    plt.tight_layout()
    plt.show(block=block)

def read_data(filepath):
    metadata = None

    data = np.array(list(map(list, np.genfromtxt(filepath, dtype=None, skip_header=1, encoding=None))))

    inputs = data[:,:-1].astype(float)
    labels = data[:,-1].astype(str)

    _, labels = np.unique(labels, return_inverse=True)
    
    return inputs.T, labels

def get_hyperparameter_configurations(json_filepath):
    data = None

    with open(json_filepath) as f:
        data = json.load(f)

    return data

def plot_confusion_matrix(test_labels, test_predicted, n_classes=3, block=False):
    conf_mat = confusion_matrix(test_labels, test_predicted)

    alphabets = tuple(string.ascii_lowercase)[:n_classes]

    fig, ax = plt.subplots()
    cax = ax.matshow(conf_mat, cmap='Blues')
    fig.colorbar(cax)

    plt.xticks(np.arange(n_classes), alphabets)
    plt.yticks(np.arange(n_classes), alphabets)

    for (i, j), z in np.ndenumerate(conf_mat):
        ax.text(j, i, '{:0.1f}'.format(z), ha='center', va='center')

    plt.show(block=False)

    plt.savefig('test_confusion_matrix.png')
