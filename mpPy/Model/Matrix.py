#!/usr/bin/python
# -*- coding: utf-8 -*-

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
    def __init__(self, matrix):
        self._orig_matrix = matrix
        self._instances = self._orig_matrix.shape[0]
        self._dimensions = (self._orig_matrix.shape[1] - 1)
        self._clusters = self._orig_matrix[:, (self._dimensions)]
        self._initial_2D_matrix = np.random.random((self._instances, 2))
        self._data_matrix = self._orig_matrix.copy()

    def data_matrix(self):
        return self._data_matrix[:, range(self.dimensions())]

    def orig_matrix(self):
        return self._orig_matrix

    def instances(self):
        return self._instances

    def dimensions(self):
        return self._dimensions

    def clusters(self):
        return self._clusters.astype(int)

    def initial_2D_matrix(self):
        return self._initial_2D_matrix

    def __str__(self):
        return '<{}: {} - {} >\n'.format(self.__class__.__name__, self.instances(), self.dimensions())

class Reader(object):
    def __init__(self):
        pass

    def reader_file(self, file_name):
        try:
            return np.loadtxt(file_name, delimiter=",")
        except IOError as e:
            print("Não foi possível abrir o arquivo", file_name)
            print(e)
