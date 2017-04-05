#!/usr/bin/python
# -*- coding: utf-8 -*-

import numpy as np

class Matrix():
    def __init__(self, matrix):
        self.orig_matrix = matrix
        self.instances = self.orig_matrix.shape[0]
        self.dimensions = (self.orig_matrix.shape[1] - 1)
        self.clusters = self.orig_matrix[:, self.dimensions]
        self.initial_2D_matrix = np.random.random((self.instances, 2))
        self.data_matrix = matrix[:,:-1]

    def clusters(self):
        return self.clusters.astype(int)

    def __str__(self):
        return '<Técnica {}: {} instâncias - {} atributos >\n'.format(self.__class__.__name__, self.instances(), self.dimensions())

class Reader(object):
    def __init__(self):
        pass

    def reader_file(self, file_name):
        try:
            return np.loadtxt(file_name, delimiter=",")
        except IOError as e:
            print("Não foi possível abrir o arquivo", file_name)
            print(e)

    def reader_file2(self, file_name):
        try:
            return np.genfromtxt(file_name, delimiter=";")
        except IOError as e:
            print("Não foi possível abrir o arquivo", file_name)
            print(e)