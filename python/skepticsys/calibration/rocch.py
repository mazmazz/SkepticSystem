from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.utils import indexable, column_or_1d
from sklearn.metrics import roc_curve
import numpy as np

class ConvexHull():
    """ Private function that calculate the convex hull of a set of 2-D points
    The algorithm was taken from [1].
    http://code.activestate.com/recipes/66527-finding-the-convex-hull-of-a-set-of-2d-points/

    Convex hulls in 2 dimensions.

    Parameters
    ----------
    points : ndarray of floats, shape (npoints, 2)
        Coordinates of points to construct a convex hull from

    Attributes
    ----------
    vertices : ndarray of ints, shape (nvertices,)
        Indices of points forming the vertices of the convex hull.
        For 2-D convex hulls, the vertices are in counterclockwise order.
        For other dimensions, they are in input order.

    References
    ----------
    .. [1] Alex Martelli, Anna Ravenscroft, David Ascher, 'Python Cookbook', O'Reilly Media, Inc., 2005.

    """

    def mydet(self, p, q, r):
        """Calc. determinant of a special matrix with three 2D points.

        The sign, "-" or "+", determines the side, right or left,
        respectivly, on which the point r lies, when measured against
        a directed vector from p to q.
        """

        # We use Sarrus' Rule to calculate the determinant.
        # (could also use the Numeric package...)
        sum1 = q[0] * r[1] + p[0] * q[1] + r[0] * p[1]
        sum2 = q[0] * p[1] + r[0] * q[1] + p[0] * r[1]

        return sum1 - sum2

    def isrightturn(self, p, q, r):
        "Do the vectors pq:qr form a right turn, or not?"

        assert p != q and q != r and p != r

        if self.mydet(p, q, r) < 0:
            return 1
        else:
            return 0

    def __init__(self, points):
        # Get a local list copy of the points and sort them lexically.
        points_ = points.tolist()
        points_.sort()

        # Build upper half of the hull.
        upper = [points_[0], points_[1]]
        for p in points_[2:]:
            upper.append(p)
            while len(upper) > 2 and not self.isrightturn(*upper[-3:]):
                del upper[-2]

        # Build lower half of the hull.
        points_.reverse()
        lower = [points_[0], points_[1]]
        for p in points_[2:]:
            lower.append(p)
            while len(lower) > 2 and not self.isrightturn(*lower[-3:]):
                del lower[-2]

        # Remove duplicates.
        del lower[0]
        del lower[-1]

        # Concatenate both halfs and construct vertices.
        c_points = np.array(tuple(upper + lower))
        vertices = []
        for point in c_points:
            vertices.append(np.intersect1d(
                            np.nonzero(point[0] == points[:, 0])[0],
                            np.nonzero(point[1] == points[:, 1])[0])[0])

        self.vertices = np.array(vertices)

class _ROCCHCalibration(BaseEstimator, RegressorMixin):
    """Implementation the the calibration method ROCConvexHull

    Attributes
    ----------
    `calibration_map` : array-like
        calibration map for mapping the raw probabilities to the calibrated probabilities.

    References
    ----------
    .. [5] A Unified View of Performance Metrics : Translating Threshold Choice
           into Expected Classification Loss. J. Hernandez-Orallo & P. Flach &
           C. Ferri. JMLR 2012.

    """
    def fit(self, df, y, sample_weight=None):
        """Fit the model using X, y as training data.

        Parameters
        ----------
        df : ndarray, shape (n_samples,)
            The decision function or predict proba for the samples.

        y : ndarray, shape (n_samples,)
            The targets.

        sample_weight : array-like, shape = [n_samples] or None
            Sample weights. If None, then samples are equally weighted.

        Returns
        -------
        self : object
            Returns an instance of self.
        """

        df = column_or_1d(df)
        y = column_or_1d(y)
        df, y = indexable(df, y)

        # Calculate the roc curve
        fpr, tpr, thresholds = roc_curve(y, df, sample_weight=sample_weight)

        # Calculate convex hull
        temp_ch = np.vstack((fpr,tpr)).T
        # Close the plane to avoid issues when points are on the right of the diag
        temp_ch = np.r_[temp_ch, np.array([[1, 0]])]
        hull = ConvexHull(temp_ch)
        hull_idx = np.sort(hull.vertices)[:-1]

        # Get new thresholds and invert order
        ch_thresholds = thresholds[hull_idx]
        ch_thresholds = ch_thresholds[::-1]

        # Fix to generalize
        ch_thresholds[0] = -np.inf
        ch_thresholds[-1] = np.inf

        # Distribution between new thresholds
        binids = np.digitize(df, ch_thresholds) - 1

        bin_true = np.bincount(binids, weights=y, minlength=len(ch_thresholds))
        bin_total = np.bincount(binids, minlength=len(ch_thresholds))

        nonzero = bin_total != 0
        prob_true = (bin_true[nonzero] / bin_total[nonzero])

        self.calibration_map = (ch_thresholds, prob_true)

        return self

    def predict(self, T):
        """Predict new data by applying the calibration map

        Parameters
        ----------
        T : array-like, shape (n_samples,)
            Data to predict from.

        Returns
        -------
        `T_` : array, shape (n_samples,)
            The predicted data.
        """
        T = column_or_1d(T)

        ch_thresholds, prob_true = self.calibration_map

        T_ = np.digitize(T, ch_thresholds) - 1

        return prob_true[T_]
