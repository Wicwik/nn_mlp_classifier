import numpy as np

from classifier import *
from util import *

data_train = np.genfromtxt('2d.trn.dat', dtype=None, encoding=None)
data_test = np.genfromtxt('2d.tst.dat', dtype=None, encoding=None)

print(data_train)