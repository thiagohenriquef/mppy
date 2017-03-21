#!/usr/bin/python
#-*- coding: utf-8 -*-

try:
    import numpy as np
    import scipy as sp
    import sklearn as sk
except ImportError as e:
    print("Please install the following packages: ")
    print("Numpy: http://www.numpy.org/")
    print("Scipy: https://www.scipy.org/")
    print("Scikit Learn: http://scikit-learn.org/stable/")


class Matrix():

    def __init__(self, data_matrix):
        self.data_matrix = data_matrix
        self.instances = data_matrix.shape[0]
        self.dimensions = (data_matrix.shape[1]-1)
        self.clusters = data_matrix[:, (self.dimensions-1)]
        self.initial_2D_matrix = np.random.random((self.instances, 2))

    def __str__(self):
        return "Abstract Matrix initial"