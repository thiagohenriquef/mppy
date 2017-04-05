#!/usr/bin/python
#-*- coding: utf-8 -*-

import matplotlib.pyplot as mplpy

class Plot():
    def __init__(self):
        pass

    def __init__(self, bidimensional_representation, clusters, matrix):
        self.bidimensional_representation = bidimensional_representation
        self.clusters = clusters
        self.matrix = matrix

    def simple_scatter_plot(self):
        mat_2D = self.bidimensional_representation
        clusters = self.clusters
        mplpy.scatter(mat_2D[:,0], mat_2D[:,1], c=clusters)
        mplpy.show()

    def semi_interactive_scatter_plot(self):
        mat_2D = self.bidimensional_representation
        clusters = self.clusters
        matrix = self.matrix

        if 1:  # picking on a scatter plot (matplotlib.collections.RegularPolyCollection)
            def onpick3(event):
                ind = event.ind
                print("Elemento: ",ind,"->", matrix[ind,:])

            fig = mplpy.figure()
            ax1 = fig.add_subplot(111)
            ax1.scatter(mat_2D[:,0], mat_2D[:,1], c=clusters, picker=True)
            fig.canvas.mpl_connect('pick_event', onpick3)

        mplpy.show()