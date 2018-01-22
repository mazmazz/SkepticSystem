from .classifiercv import ClassifierCV

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator
from sklearn.metrics import roc_curve
from sklearn.utils.validation import check_is_fitted
from sklearn.metrics.pairwise import euclidean_distances
import logging

# parent submodules
import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils import is_pandas, get_proba
sys.path.pop(0)
# end parent submodules

class ThresholdClassifierCV(ClassifierCV):
    def __init__(self, base_estimator=None, method='max_tpr', pos_label=1, neg_label=0, cv=3, min_val_tnr=0, min_val_tpr=0, prefit_callback=None, prefit_params = {}, postfit_callback = None, postfit_params = {}):
        self.method = method
        self.pos_label = pos_label
        self.neg_label = neg_label
        self.min_val_tnr = min_val_tnr
        self.min_val_tpr = min_val_tpr
        super().__init__(
            base_estimator=base_estimator
            , cv=cv
            , super_class=_ThresholdClassifier
            , super_params={'method':self.method,'pos_label':self.pos_label,'neg_label':self.neg_label,'min_val_tnr':self.min_val_tnr,'min_val_tpr':self.min_val_tpr}
            , prefit_callback=prefit_callback
            , prefit_params=prefit_params
            , postfit_callback=postfit_callback
            , postfit_params=postfit_params
        )
        self.thresholds_ = None

    @property
    def thresholds(self):
        if self.thresholds_ is None:
            self.thresholds_ = [check_threshold(cl.threshold_) for cl in self.classifiers_]
        return self.thresholds_

    @property
    def threshold(self):
        return check_threshold(sum(self.thresholds) / len(self.thresholds))

    def reset_cv(self):
        super().reset_cv()
        self.thresholds_ = None

    def predict(self, X):
        check_is_fitted(self, ["classes_", "classifiers_"])
        return (get_proba(self.predict_proba(X), proba_positive=True) >= self.threshold).astype(int)

# https://github.com/scikit-learn/scikit-learn/pull/10117 from Jan 15 2018 (6f9ce4ac7337e448c6619a45c6136ceb1c6297d3)

class _ThresholdClassifier(BaseEstimator):
    """Optimal threshold point selection.

    It assumes that base_estimator has already been fit, and uses the input set
    of the fit function to select an optimal threshold point. Note that this
    class should not be used as an estimator directly. Use the
    OptimalCutoffClassifier with cv="prefit" instead.

    Parameters
    ----------
    base_estimator : instance BaseEstimator
        The classifier whose decision threshold will be adapted according to
        the acquired optimal threshold point

    method : 'roc' or 'max_tpr' or 'max_tnr'
        The method to use for choosing the threshold point.

    pos_label : object
        Label considered as positive during the roc_curve construction.

    min_val_tnr : float in [0, 1]
        minimum required value for true negative rate (specificity) in case
        method 'max_tpr' is used

    min_val_tpr : float in [0, 1]
        minimum required value for true positive rate (sensitivity) in case
        method 'max_tnr' is used

    Attributes
    ----------
    threshold_ : float
        Acquired optimal decision threshold for the positive class
    """
    def __init__(self, base_estimator, method, pos_label, neg_label, min_val_tnr,
                 min_val_tpr):
        self.base_estimator = base_estimator
        self.method = method
        self.pos_label = pos_label
        self.neg_label = neg_label
        self.min_val_tnr = min_val_tnr
        self.min_val_tpr = min_val_tpr

    def fit(self, X, y):
        """Select a decision threshold for the fitted model's positive class
        using one of the available methods

        Parameters
        ----------
        X : array-like, shape (n_samples, n_features)
            Training data.

        y : array-like, shape (n_samples,)
            Target values.

        Returns
        -------
        self : object
            Instance of self.
        """
        y_score = self.base_estimator.predict_proba(X)[:, self.pos_label]
        fpr, tpr, thresholds = roc_curve(y, y_score, self.pos_label)

        if self.method == 'roc':
            self.threshold_ = _topleft_threshold(y_score, y, pos_label=self.pos_label)
        elif self.method == 'youden':
            self.threshold_ = _youden_threshold(y_score, y, pos_label=self.pos_label)
        elif self.method == 'max_tpr':
            self.threshold_ = _max_tpr_threshold(y_score, y, self.min_val_tpr, pos_label=self.pos_label)
        elif self.method == 'max_tnr':
            self.threshold_ = _max_tnr_threshold(y_score, y, self.min_val_tnr, pos_label=self.pos_label)
        else:
            raise ValueError('method must be "roc" or "youden" or "max_tpr" or "max_tnr.'
                             'Got %s instead' % self.method)
        return self

    def predict_proba(self, X):
        return self.base_estimator.predict_proba(X)

    def predict(self, X):
        return self.pos_label if get_proba(self.base_estimator.predict_proba(X), proba_positive=True) >= self.threshold_ else self.neg_label

def _max_tnr_threshold(df, y, min_val_tnr, pos_label=1):
    fpr, tpr, thresholds = roc_curve(y, y_score, pos_label=pos_label)

    if not min_val_tnr >= 0 or not min_val_tnr <= 1:
        raise ValueError('max_tnr must be a number in [1, 0]. '
                            'Got %s instead' % repr(min_val_tnr))
    indices = np.where(1 - fpr >= min_val_tnr)[0]
    return thresholds[indices[np.argmax(tpr[indices])]]

def _max_tpr_threshold(df, y, min_val_tpr, pos_label=1):
    fpr, tpr, thresholds = roc_curve(y, df, pos_label=pos_label)

    if not min_val_tpr >= 0 or not min_val_tpr <= 1:
        raise ValueError('max_tpr must be a number in [1, 0]. '
                            'Got %s instead' % repr(min_val_tpr))
    indices = np.where(tpr >= min_val_tpr)[0]
    return thresholds[indices[np.argmax(1 - fpr[indices])]]

def _youden_threshold(df, y, pos_label=1):
    """Find the optimal probability threshold point for a classification model related to event rate using Youden index

    Parameters
    ----------
    df: 
        Decision function or predicted probabilities

    y: 
        The targets

    Returns
    -------     
    list type, with optimal cutoff value
    """
    # https://stackoverflow.com/questions/28719067/roc-curve-and-cut-off-point-python
    fpr, tpr, threshold = roc_curve(y, df, pos_label=pos_label, drop_intermediate=False)
    i = np.arange(len(tpr)) 
    roc = pd.DataFrame({'tf' : pd.Series(tpr-(1-fpr), index=i), 'threshold' : pd.Series(threshold, index=i)})
        # youden = max(sensitivity+specificity) or max(tpr+tnr) or max(tpr+(1-fpr))
    roc_t = roc.ix[(roc.tf-0).abs().argsort()[:1]]
    
    return roc_t['threshold'].iloc[0] # list(roc_t['threshold'])

def _topleft_threshold(df, y, pos_label=1):
    """Find the optimal probability threshold point for a classification model from ROC curve using point closest to (0,1)

    Parameters
    ----------
    df:
        Decision function or predicted probabilities

    y:
        The targets

    Returns
    -------     
    list type, with optimal cutoff value
    """
    fpr, tpr, thresholds = roc_curve(y, df, pos_label=pos_label, drop_intermediate=False) #, self.pos_label) 
    
    if np.any(np.isnan(tpr)) or np.any(np.isnan(fpr)):
        return np.nan
    else:
        return thresholds[np.argmin( 
            euclidean_distances(np.column_stack((fpr, tpr)), [[0, 1]]) 
        )] 

def check_threshold(thr):
    if thr is None:
        raise ValueError('Threshold is not fitted, call fit() first.')
    elif thr > 1 or thr < 0:
        raise ValueError('Threshold %.8f is not between 0 and 1.' % (thr))
    elif np.isnan(thr):
        logging.getLogger().warn('Threshold is NAN, treating as 0.5')
        threshold = 0.5
    else:
        threshold = thr
    
    return threshold
