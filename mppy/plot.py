def simple_scatter_plot(matrix_2d, clusters=None):
    """
    Basic implementation of a scatter plot.

    :param matrix_2d: ndarray(m,2)
        The matrix 2d that will be plotted.
    :param clusters: ndarray(m, )
        Clustering of the dataset with your respectives classes. If clusters is
        None, the matrix will be projected with only one color.
    :return: None
    """
    import matplotlib.pyplot as mplpy
    import numpy as np

    if clusters is None:
        clusters = np.zeros((matrix_2d.shape[0]))
    mplpy.scatter(matrix_2d[:, 0], matrix_2d[:, 1],
                  c=clusters.astype(int), marker='o', alpha=1, edgecolors='black')
    mplpy.show()


def interactive_scatter_plot(matrix_2d, matrix, clusters=None):
    """
    Implementation of an interactive scatter plot with the mouse click.

    :param matrix_2d: ndarray(m,2)
        The matrix 2d that will be plotted.
    :param matrix: ndarray(m,n)
        The original multidimensional dataset.
    :param clusters: ndarray(m, )
        Clustering of the dataset with your respectives classes. If clusters is
        None, the matrix will be projected with only one color.
    :return: None
    """
    import matplotlib.pyplot as mplpy
    import numpy as np

    if clusters is None:
        clusters = np.zeros((matrix_2d.shape[0]))

    if 1:  # picking on a scatter plot (matplotlib.collections.RegularPolyCollection)

        def onpick3(event):
            ind = event.ind
            print("Element: ", ind, "->", matrix[ind, :])

        fig = mplpy.figure()
        ax1 = fig.add_subplot(111)
        ax1.scatter(matrix_2d[:, 0], matrix_2d[:, 1], c=clusters.astype(int), picker=True, marker="o", alpha=1,
                    edgecolors='black')
        fig.canvas.mpl_connect('pick_event', onpick3)

    mplpy.show()
