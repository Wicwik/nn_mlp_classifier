# Neural Networks (2-AIN-132/15), FMFI UK BA
# (c) Tomas Kuzma, Juraj Holas, Peter Gergel, Endre Hamerlik, Štefan Pócoš, Iveta Bečková 2017-2022

import matplotlib
matplotlib.use('TkAgg') # fixme if plotting doesn`t work (try 'Qt5Agg' or 'Qt4Agg')
import matplotlib.pyplot as plt
# for 3D visualization
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401 unused import
from matplotlib import cm

import numpy as np
import atexit
import os
import time
import functools



## Utilities

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
def plot_errors(title, errors, test_error=None, block=True):
    plt.figure()
    use_keypress()

    plt.plot(errors)
    plt.ylim(bottom=0)

    if test_error:
        plt.plot([test_error]*len(errors))

    plt.tight_layout()
    plt.gcf().canvas.set_window_title(title)
    plt.show(block=block)


def plot_reg_density(title, inputs, targets, outputs=None, s=70, block=True, plot_3D=True):
    fig = plt.figure(figsize=(9,9))
    use_keypress()

    if plot_3D:
        img_dim = (30, 20)

        X = inputs[0].reshape(img_dim)
        Y = inputs[1].reshape(img_dim)
        T = targets.reshape(img_dim)

        ax_orig = fig.gca(projection='3d')
        ax_orig.plot_surface(X, Y, T, cmap=cm.coolwarm, linewidth=0, antialiased=True)
        plt.gcf().canvas.set_window_title('Original')

        if outputs is not None:
            O = outputs.reshape(img_dim)

            fig_pred = plt.figure(figsize=(9,9))
            use_keypress(fig_pred)
            ax_pred = fig_pred.gca(projection='3d')
            ax_pred.plot_surface(X, Y, O, cmap=cm.coolwarm, linewidth=0, antialiased=True)
            plt.gcf().canvas.set_window_title('Predicted')
    else:
        trg = np.maximum(0, np.array(targets).flatten())
        vmax = np.max(targets)

        if outputs is not None:
            out = np.maximum(0, np.array(outputs).flatten())
            c = np.zeros(inputs[0].shape) if len(out) == 1 else out
            vmax = np.max((vmax, np.max(out)))
            plt.subplot(2,1,2)
            plt.title('Predicted')
            plt.scatter(inputs[0], inputs[1], s=s*out, c=c, cmap='viridis', vmax=vmax)
            plt.colorbar()

            plt.subplot(2,1,1)
            plt.title('Original')

        plt.scatter(inputs[0], inputs[1], s=s*trg, c=trg, cmap='viridis', vmax=vmax)
        plt.colorbar()
        plt.gcf().canvas.set_window_title(title)

    plt.tight_layout()

    plt.show(block=block)
