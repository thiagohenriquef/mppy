import numpy as np
from scipy.spatial.distance import pdist, squareform


def calculate_kruskal_stress(distance_rn, distance_r2):
    """
    distance_r2 = squareform(pdist(distance_r2))
    distance_rn = squareform(pdist(distance_rn))
    num = np.sum(np.power(distance_r2 - distance_rn, 2))
    den = np.sum(np.power(distance_rn, 2))
    """

    distance_rn = squareform(pdist(distance_rn))
    distance_r2 = squareform(pdist(distance_r2))

    num = 0.0
    den = 0.0
    for i in range(distance_rn.shape[0]):
        for j in range(1, distance_r2.shape[0]):
            if i < j:
                dist_rn = distance_rn[i,j]
                dist_r2 = distance_r2[i,j]

                num += (dist_rn - dist_r2) * (dist_rn - dist_r2)
                den += dist_rn * dist_rn

    return num / den