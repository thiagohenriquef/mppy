from scipy.spatial.distance import squareform, pdist
from mpPy.Model.Matrix import Matrix
import numpy as np
import traceback

class KruskalStress(Matrix):
    """
    Kruskal Stress
    """
    def __init__(self, Matrix):
        self.distance_rn = squareform(pdist(Matrix.data_matrix))
        self.distance_r2 = squareform(pdist(Matrix.initial_2D_matrix))

    def calculate(self):
        num = 0.0
        den = 0.0

        num = np.sum(np.power(self.distance_r2 - self.distance_rn, 2))
        den = np.sum(np.power(self.distance_rn, 2))

        return num / den