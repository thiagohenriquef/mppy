import numpy as np
from scipy.spatial.distance import pdist

def calculate_kruskal_stress(distance_rn, distance_r2):
    num = np.sum(np.power(pdist(distance_r2) - pdist(distance_rn), 2))
    den = np.sum(np.power(pdist(distance_rn), 2))

    return num / den