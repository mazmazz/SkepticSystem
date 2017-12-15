from sklearn.base import BaseEstimator, RegressorMixin
import numpy as np

def _euler_sigmoid_calibration(x):
    """For applying binary:logistic to raw xgboost scores"""
    # https://github.com/dmlc/xgboost/blob/40c6e2f0c84db46070f91cf77db03275a1f4eee9/src/objective/regression_obj.cc#L55
    # https://github.com/dmlc/xgboost/blob/40c6e2f0c84db46070f91cf77db03275a1f4eee9/src/common/math.h#L22
    return 1.0/(1.0+np.exp(-x))

class EulerSigmoidCalibration(BaseEstimator, RegressorMixin):
    """Transformer classifier scores using sigmoid and exp.
    """
    def __init__(self):
        pass

    def fit(self, X, y, sample_weight=None):
        """Stub function.

        Returns
        -------
        self : object
            Returns an instance of self.
        """
        return self

    def predict(self, S):
        """Predict new values.

        Parameters
        ----------
        S : array-like, shape (n_samples,)
            Data to predict from.

        Returns
        -------
        : array, shape (n_samples,)
            The predicted values.
        """
        return _euler_sigmoid_calibration(S)
