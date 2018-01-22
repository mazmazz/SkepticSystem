from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.utils import column_or_1d
import numpy as np

def _euler_sigmoid_calibration(x):
    """For applying binary:logistic to raw xgboost scores"""
    # https://github.com/dmlc/xgboost/blob/40c6e2f0c84db46070f91cf77db03275a1f4eee9/src/objective/regression_obj.cc#L55
    # https://github.com/dmlc/xgboost/blob/40c6e2f0c84db46070f91cf77db03275a1f4eee9/src/common/math.h#L22
    return 1.0/(1.0+np.exp(-x))

class _EulerSigmoidCalibration(BaseEstimator, RegressorMixin):
    """Transformer classifier scores using sigmoid and exp.
    """
    def __init__(self):
        pass

    def fit(self, X, y, sample_weight=None):
        """Stub function, no fitting needed.

        Returns
        -------
        self : object
            Returns an instance of self.
        """
        return self

    def predict(self, T):
        """Predict new values.

        Parameters
        ----------
        T : array-like, shape (n_samples,)
            Data to predict from.

        Returns
        -------
        : array, shape (n_samples,)
            The predicted values.
        """
        T = column_or_1d(T)
        return _euler_sigmoid_calibration(T)
