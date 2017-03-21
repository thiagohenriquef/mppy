from Matrix import Matrix

class Force(Matrix):
    """
    Force Scheme Projection

    """

    def __init__(self, max_iterations=50, tolerance=0.0, fraction_of_delta=8.0, epsilon=1e-6):
        super.__init__()
        self.max_iterations = max_iterations
        self.tolerance = tolerance
        self.fraction_of_delta = fraction_of_delta
        self.epsilon = epsilon

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

