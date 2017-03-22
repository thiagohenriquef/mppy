from mpPy.Model.Matrix import Matrix


class ForceScheme(Matrix):
    """
    Force Scheme Projection

    """

    def __init__(self, matrix, max_iterations=50, tolerance=0.0, fraction_of_delta=8.0, epsilon=1e-6):
        super().__init__(matrix)
        self._max_iterations = max_iterations
        self._tolerance = tolerance
        self._fraction_of_delta = fraction_of_delta
        self._epsilon = epsilon

    def __getattribute__(self, *args, **kwargs):
        return super().__getattribute__(*args, **kwargs)

    def max_iterations(self):
        return self._max_iterations

    def tolerance(self):
        return self._tolerance

    def fraction_of_delta(self):
        return self._fraction_of_delta

    def epsilon(self):
        return self._epsilon


class LSP(Matrix):
    """

    Least Square Projeciton

    """

    def __init__(self):
        super.__init__()
        self.subsample_indices = None
        self.num_neighbors = None
        self.initial_neighbors = None
        self.dimensionality = None


class Pekalska(Matrix):
    """

    Pekalska Approximation

    """

    def __init__(self):
        self.subsample_indices = None
        self.subsample_mapping = None


class PLMP(Matrix):
    """

    Part-Linear Multidimensional Projection

    """

    def __init__(self):
        self.subsample_indices = None
        self.subsample_control_points = None


class LAMP(Matrix):
    """

    Local Affine Multidimensional Projection

    """

    def __init__(self):
        self.subsample_indices = None
        self.initial_2d_subsample = None
        self.proportion = None
