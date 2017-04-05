from mppy.model.matrix import Matrix

class Sammon(Matrix):
    """
    Sammon Approach
    """
    def __init__(self, matrix,
                 num_iterations = 50,
                 magic_factor = 0.03,
                 tolerance = 0.0):
        super().__init__(matrix)
        self.num_iterations = num_iterations
        self.magic_factor = magic_factor
        self.tolerance = tolerance