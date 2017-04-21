def kruskal_stress(distance_rn, distance_r2):
    """

    :param distance_rn:
    :param distance_r2:
    :return:
    """

    from scipy.spatial.distance import pdist, squareform

    """
    distance_r2 = squareform(pdist(distance_r2))
    distance_rn = squareform(pdist(distance_rn))
    num = np.sum(np.power(distance_r2 - distance_rn, 2))
    den = np.sum(np.power(distance_rn, 2))

    """
    distance_rn = squareform(pdist(distance_rn), 'euclidean')
    distance_r2 = squareform(pdist(distance_r2), 'euclidean')

    num = 0.0
    den = 0.0
    for i in range(distance_rn.shape[0]):
        for j in range(1, distance_rn.shape[0]):
            dist_rn = distance_rn[i, j]
            dist_r2 = distance_r2[i, j]

            num += (dist_rn - dist_r2) * (dist_rn - dist_r2)
            den += dist_rn * dist_rn

    return num / den

def normalized_kruskal_stress(distance_rn, distance_r2):
    """

    :param distance_rn:
    :param distance_r2:
    :return:
    """

    from scipy.spatial.distance import pdist, squareform
    import math
    distance_rn = squareform(pdist(distance_rn), 'euclidean')
    distance_r2 = squareform(pdist(distance_r2), 'euclidean')

    max_rn = -math.inf
    max_r2 = -math.inf

    for x in range(distance_rn.shape[0]):
        for y in range(1, distance_rn.shape[0]):
            value_rn = distance_rn[x,y]
            value_r2 = distance_r2[x,y]

            if value_r2 > max_r2:
                max_r2 = value_r2

            if value_rn > max_rn:
                max_rn = value_rn

    num = 0.0
    den = 0.0
    for i in range(distance_rn.shape[0]):
        for j in range(1, distance_rn.shape[0]):
            dist_rn = distance_rn[i, j] / max_rn
            dist_r2 = distance_r2[i, j] / max_r2

            num += (dist_rn - dist_r2) * (dist_rn - dist_r2)
            den += dist_rn * dist_rn

    return num / den