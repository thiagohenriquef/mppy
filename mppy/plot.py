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

    mplpy.style.use('classic')
    mplpy.scatter(matrix_2d[:, 0], matrix_2d[:, 1],
                  c=clusters.astype(int), s=40, marker='o', alpha=1, edgecolors='black')
    mplpy.show()
    mplpy.close()


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
        ax1.scatter(matrix_2d[:, 0], matrix_2d[:, 1], s=40, c=clusters.astype(int), picker=True, marker="o", alpha=1,
                    edgecolors='black')
        fig.canvas.mpl_connect('pick_event', onpick3)

    mplpy.show()
    mplpy.close()

def delaunay_scatter(matrix_2d, data, clusters):
    from scipy.spatial import Delaunay
    import matplotlib.pyplot as plt
    tri = Delaunay(matrix_2d)
    import scipy 
    scipy.spatial.delaunay_plot_2d(tri)
    plt.triplot(matrix_2d[:, 0], matrix_2d[:, 1], tri.simplices.copy())
    plt.scatter(matrix_2d[:, 0], matrix_2d[:, 1],
                  c=clusters.astype(int), s=40,marker='o', alpha=1, edgecolors='black')
    plt.show()
    plt.close()


def interactive_scatter_plot2(matrix_2d, matrix, clusters=None):
    import matplotlib.pyplot as plt
    import numpy as np
    import scipy.spatial as spatial
    plt.style.use('classic')

    def fmt(x, y):
        a = np.array([x,y])
        d = np.where(matrix_2d == a)[0]
        c = str(d[0]+1)
        b = str(matrix[d,:][0])
        string = "Index: "+c+"-->"+b
        return string
        #return 'd: x: {x:0.2f}\ny: {y:0.2f}'.format(d=d[0], x=x, y=y)
        #return "Element: ", ind, "->", matrix[ind, :]

    class FollowDotCursor(object):
        """Display the x,y location of the nearest data point."""
        def __init__(self, ax, x, y, tolerance=5, formatter=fmt, offsets=(-20, 20)):
            try:
                x = np.asarray(x, dtype='float')
            except (TypeError, ValueError):
                x = np.asarray(mdates.date2num(x), dtype='float')
            y = np.asarray(y, dtype='float')
            self._points = np.column_stack((x, y))
            self.offsets = offsets
            self.scale = x.ptp()
            self.scale = y.ptp() / self.scale if self.scale else 1
            self.tree = spatial.cKDTree(self.scaled(self._points))
            self.formatter = formatter
            self.tolerance = tolerance
            self.ax = ax
            self.fig = ax.figure
            self.ax.xaxis.set_label_position('top')
            self.dot = ax.scatter(
                [x.min()], [y.min()], s=130, color='green', alpha=0.7)
            self.annotation = self.setup_annotation()
            plt.connect('motion_notify_event', self)

        def scaled(self, points):
            points = np.asarray(points)
            return points * (self.scale, 1)

        def __call__(self, event):
            ax = self.ax
            # event.inaxes is always the current axis. If you use twinx, ax could be
            # a different axis.
            if event.inaxes == ax:
                x, y = event.xdata, event.ydata
            elif event.inaxes is None:
                return
            else:
                inv = ax.transData.inverted()
                x, y = inv.transform([(event.x, event.y)]).ravel()
            annotation = self.annotation
            x, y = self.snap(x, y)
            annotation.xy = x, y
            annotation.set_text(self.formatter(x, y))
            self.dot.set_offsets((x, y))
            bbox = ax.viewLim
            event.canvas.draw()

        def setup_annotation(self):
            """Draw and hide the annotation box."""
            annotation = self.ax.annotate(
                '', xy=(0, 0), ha='right',
                xytext=self.offsets, textcoords='offset points', va='bottom',
                bbox=dict(
                    boxstyle='round,pad=0.5', fc='blue', alpha=0.75),
                arrowprops=dict(
                    arrowstyle='->', connectionstyle='arc3,rad=0'))
            return annotation

        def snap(self, x, y):
            """Return the value in self.tree closest to x, y."""
            dist, idx = self.tree.query(self.scaled((x, y)), k=1, p=1)
            try:
                return self._points[idx]
            except IndexError:
                # IndexError: index out of bounds
                return self._points[0]

    fig = plt.figure()
    ax = fig.add_subplot(1, 1, 1)
    ax.scatter(matrix_2d[:, 0], matrix_2d[:, 1], s=40,
                  c=clusters.astype(int), marker='o', alpha=1, edgecolors='black')
    cursor = FollowDotCursor(ax, matrix_2d[:,0], matrix_2d[:,1])
    plt.show()
    plt.close()
