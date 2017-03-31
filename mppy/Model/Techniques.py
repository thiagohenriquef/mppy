import numpy

from mppy.Model.Matrix import Matrix

class ForceScheme(Matrix):
    """
    Force Scheme Projection

    """

    def __init__(self, matrix,
                 max_iterations=50,
                 tolerance=0.0,
                 fraction_of_delta=8.0,
                 epsilon=1e-6):
        super().__init__(matrix)
        self.max_iterations = max_iterations
        self.tolerance = tolerance
        self.fraction_of_delta = fraction_of_delta
        self.epsilon = epsilon

class LSP(Matrix):
    """

    Least Square Projeciton

    """

    def __init__(self, matrix,
                 sample_indices = None,
                 sample_project=None,
                 num_neighbors = 15,
                 dimensionality = 2):

        super().__init__(matrix)
        self.sample_indices = sample_indices
        self.sample_project = sample_project
        self.num_neighbors = num_neighbors
        self.dimensionality = dimensionality

class Pekalska(Matrix):
    """

    Pekalska Approximation

    """
    def __init__(self, matrix,
                 subsample_indices = None,
                 sample_data = None):
        super().__init__(matrix)
        self.subsample_indices = subsample_indices
        self.sample_data = sample_data

class PLMP(Matrix):
    """

    Part-Linear Multidimensional Projection

    """
    def __init__(self, matrix,
                 sample_indices = None,
                 sample_data = None,
                 dimensionality = 2):
        super().__init__(matrix)
        self.sample_indices = sample_indices
        self.sample_data = sample_data
        self.dimensionality = dimensionality

class LAMP(Matrix):
    """

    Local Affine Multidimensional Projection

    """

    def __init__(self, matrix,
                 proportion = 1,
                 subsample_indices = None,
                 initial_sample = None):
        super().__init__(matrix)
        self.subsample_indices = subsample_indices
        self.initial_sample = initial_sample
        self.proportion = proportion

class Sammon(Matrix):
    """
    Piecewise Laplacian Projection
    """
    def __init__(self, matrix,
                 num_iterations = 50,
                 magic_factor = 0.03,
                 tolerance = 0.0):
        super().__init__(matrix)
        self.num_iterations = num_iterations
        self.magic_factor = magic_factor
        self.tolerance = tolerance