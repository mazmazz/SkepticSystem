import pandas as pd
import numpy as np
from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.metrics import roc_curve, precision_score, accuracy_score
from sklearn.metrics.pairwise import euclidean_distances
from sklearn.preprocessing import MinMaxScaler
from sklearn.utils import column_or_1d, indexable
import logging

def _youden_threshold(df, y):
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
    fpr, tpr, threshold = roc_curve(y, df, drop_intermediate=False)
    i = np.arange(len(tpr)) 
    roc = pd.DataFrame({'tf' : pd.Series(tpr-(1-fpr), index=i), 'threshold' : pd.Series(threshold, index=i)})
        # youden = max(sensitivity+specificity) or max(tpr+tnr) or max(tpr+(1-fpr))
    roc_t = roc.ix[(roc.tf-0).abs().argsort()[:1]]
    
    return roc_t['threshold'].iloc[0] # list(roc_t['threshold'])

def _topleft_threshold(df, y):
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
    fpr, tpr, thresholds = roc_curve(y, df, drop_intermediate=False) #, self.pos_label) 
    
    if np.any(np.isnan(tpr)) or np.any(np.isnan(fpr)):
        return np.nan
    else:
        return thresholds[np.argmin( 
            euclidean_distances(np.column_stack((fpr, tpr)), [[0, 1]]) 
        )] 

def _score_cutoff(df, y, scorer, pred=None, decision_threshold=0.5, maximize_score=True, **kwargs):
    """Find the optimal gray zone cutoff point for a classification model
    by finding the maximum score or loss function on each score level.

    Parameters
    ----------
    df: 
        Decision function or predicted probabilities

    y: 
        The targets

    scorer: 
        A function that takes inputs (y_true, y_pred[, pos_label, ...]); 
        (y_true, y_score[, pos_label, ...]); or (y_true, y_pred, y_score[, pos_label, ...])
        and returns a single score.

    pred: optional
        The predicted classes. If omitted, this is figured from df and decision_threshold.

    decision_threshold: float, default=0.5
        Decision threshold to separate binary classes.

    maximize_score: boolean, default=True
        Maximize the scorer, else minimize it

    Returns
    -------     
    Cutoff score
    """
    if not callable(scorer):
        raise ValueError('Scorer must be callable.')

    def _call_scorer(scorer, df, y, pred, **kwargs):
        import inspect
        sig = inspect.getfullargspec(scorer)
        if 'pos_label' not in sig.args:
            kwargs.pop('pos_label')

        if 'y_true' in sig.args and 'y_pred' in sig.args and 'y_score' not in sig.args and 'y_prob' not in sig.args:
            return scorer(y, pred, **kwargs)
        elif 'y_true' in sig.args and ('y_score' in sig.args or 'y_prob' in sig.args) and 'y_pred' not in sig.args:
            return scorer(y, df, **kwargs)
        else:
            return scorer(y, pred, df, **kwargs)

    if pred is None:
        pred = (df >= decision_threshold).astype(int)

    upper_mask, lower_mask = (df >= decision_threshold), (df < decision_threshold)

    # get upper score
    if np.count_nonzero(upper_mask) > 0:
        upper_thresholds = _get_threshold(y[upper_mask], df[upper_mask])
        thr_test = {'threshold': [], 'score': []}
        for thr in upper_thresholds:
            thr_mask = (df >= thr)
            df_thr, y_thr, pred_thr = df[thr_mask], y[thr_mask], pred[thr_mask]
            thr_test['threshold'].append(thr)
            thr_test['score'].append(_call_scorer(scorer, df_thr, y_thr, pred_thr, pos_label=1, **kwargs))

        thr_test_upper = pd.DataFrame(thr_test)
        upper_score = thr_test_upper['threshold'].loc[thr_test_upper['score'].idxmax() if maximize_score else thr_test_upper['score'].idxmin()]
    else:
        upper_score = np.nan

    # get lower score
    if np.count_nonzero(lower_mask) > 0:
        lower_thresholds = _get_threshold(y[lower_mask], df[lower_mask])
        thr_test = {'threshold': [], 'score': []}
        for thr in lower_thresholds:
            thr_mask = (df <= thr)
            df_thr, y_thr, pred_thr = df[thr_mask], y[thr_mask], pred[thr_mask]
            thr_test['threshold'].append(thr)
            thr_test['score'].append(_call_scorer(scorer, df_thr, y_thr, pred_thr, pos_label=0, **kwargs))

        thr_test_lower = pd.DataFrame(thr_test)
        lower_score = thr_test_lower['threshold'].loc[thr_test_lower['score'].idxmax() if maximize_score else thr_test_lower['score'].idxmin()]
    else:
        lower_score = np.nan

    return upper_score, lower_score

def _accuracy_count_score(y_true, y_pred, pos_label=1):
    # try precision_score?
    # accuracy_score gives cutoff with least falses possible
    # precision_score allows for some falses, but is prohibitively liberal

    counter_label = int(not bool(pos_label))
    
    acc = accuracy_score(y_true, y_pred)#, pos_label=pos_label)
    cnt = len(y_true[y_true == pos_label])
    counter_acc = accuracy_score(y_true, y_pred)#, pos_label=counter_label)
    counter_cnt = len(y_true[y_true == counter_label])

    return acc*cnt-counter_acc*counter_cnt

def _get_threshold(y_true, y_score, pos_label=1, sample_weight=None):
    # make y_true a boolean vector
    y_true = (y_true == pos_label)
    
    # sort scores and corresponding truth values
    desc_score_indices = np.argsort(y_score, kind="mergesort")[::-1]
    y_score = y_score[desc_score_indices]
    y_true = y_true[desc_score_indices]
    if sample_weight is not None:
        weight = sample_weight[desc_score_indices]
    else:
        weight = 1.

    # y_score typically has many tied values. Here we extract
    # the indices associated with the distinct values. We also
    # concatenate a value for the end of the curve.
    distinct_value_indices = np.where(np.diff(y_score))[0]
    threshold_idxs = np.r_[distinct_value_indices, y_true.size - 1]
    return y_score[threshold_idxs]

class _ThresholdCalibration(BaseEstimator, RegressorMixin):
    """Finds optimal decision threshold using either Youden index or top-left ROC point.

    Attributes
    ----------
    `_threshold` : Float
        Fitted threshold after fitting
    """

    def __init__(self, method='youden'):
        """Set parameters for threshold optimizer.

        Parameters
        ----------
        method: string, default 'youden'
            Optimization method to use: 'youden' or 'topleft'.
        """
        self.method = method
        self.threshold_ = None

    def fit(self, df, y):
        """Fit the model using X, y as training data.

        Parameters
        ----------
        df : ndarray, shape (n_samples,)
            The decision function or predict proba for the samples.

        y : ndarray, shape (n_samples,)
            The targets.

        Returns
        -------
        self : object
            Returns an instance of self.
        """
        df = column_or_1d(df)
        y = column_or_1d(y)
        df, y = indexable(df, y)

        if self.method == 'youden':
            self.threshold_ = _youden_threshold(df, y)
        elif self.method == 'topleft':
            self.threshold_ = _topleft_threshold(df, y)
        return self

    def predict(self, S):
        """Predict new values.

        Parameters
        ----------
        S : array-like, shape (n_samples,)
            Data to predict from.

        Returns
        -------
        S_ : array, shape (n_samples,)
            The predicted values.
        """
        if self.threshold_ is None:
            raise ValueError('Threshold is not fitted, call fit() first.')
        elif self.threshold_ > 1 or self.threshold_ < 0:
            raise ValueError('Threshold %.8f is not between 0 and 1.' % (self.threshold_))
        elif np.isnan(self.threshold_):
            logging.getLogger().warn('Threshold is NAN, treating as 0.5')
            threshold = 0.5
        else:
            threshold = self.threshold_
        
        scaler_0 = MinMaxScaler(feature_range=(0,0.49))
        scaler_1 = MinMaxScaler(feature_range=(0.5,1))
        idx = S.index if isinstance(S, pd.Series) or isinstance(S, pd.DataFrame) else list(range(0, len(S)))
        if not isinstance(S, pd.Series):
            S = pd.Series(column_or_1d(S), index=idx)
        S_0 = S[S < threshold]
        S_1 = S[S >= threshold]
        out_0, out_1 = None, None

        if len(S_0) > 0:
            scaler_0.fit(S.reshape(-1,1))
            out_0 = column_or_1d(scaler_0.transform(S_0.reshape(-1,1)))
        
        if len(S_1) > 0:
            scaler_1.fit(S.reshape(-1,1))
            out_1 = column_or_1d(scaler_1.transform(S_1.reshape(-1,1)))

        if out_0 is not None and out_1 is not None:
            return pd.Series().append([pd.Series(out_0, index=S_0.index), pd.Series(out_1, index=S_1.index)])[idx].as_matrix()
        elif out_0 is not None:
            return out_0 #.as_matrix()
        elif out_1 is not None:
            return out_1 #.as_matrix()
        else:
            return None

class _ThresholdClassifier(BaseEstimator, RegressorMixin):
    """Finds optimal decision threshold using either Youden index or top-left ROC point,
    then returns predicted classes based on that threshold.

    Attributes
    ----------
    `threshold_` : Float
        Fitted threshold after fitting

    calibrator_ : object, _ThresholdCalibration
        _ThresholdCalibration instance

    """

    def __init__(self, method='youden', rescale_scores=False):
        """Set parameters for threshold optimizer.

        Parameters
        ----------
        method: string, default 'youden'
            Optimization method to use: 'youden' or 'topleft'.

        rescale_scores: boolean, default False
            Rescale prediction scores to 0.5 threshold, or pass scores as-is,
            when calling predict_proba.
        """
        self.method = method
        self.rescale_scores = rescale_scores
        self.threshold_ = None
        self.calibrator_ = _ThresholdCalibration(method=method)

    def fit(self, df, y):
        """Fit the model using X, y as training data.

        Parameters
        ----------
        df : ndarray, shape (n_samples,)
            The decision function or predict proba for the samples.

        y : ndarray, shape (n_samples,)
            The targets.

        Returns
        -------
        self : object
            Returns an instance of self.
        """
        self.calibrator_.fit(df, y)
        self.threshold_ = self.calibrator_.threshold_

    def predict_proba(self, S):
        """Predict new values.

        If `rescale_scores` is True, remap scores to 0/0.5/0.1. Else, pass scores as-is.

        Parameters
        ----------
        S : array-like, shape (n_samples,)
            Data to predict from.

        Returns
        -------
        S_ : array, shape (n_samples,)
            The predicted values.
        """
        if self.rescale_scores:
            S_ = self.calibrator_.predict(S)
        else:
            S_ = column_or_1d(S)
        
        return np.column_stack((1.-S_, S_))
    
    def predict(self, S):
        """Predict classes based on threshold.

        Parameters
        ----------
        S : array-like, shape (n_samples,)
            Data to predict from.

        Returns
        -------
        S_ : array, shape (n_samples,)
            The predicted values.
        """
        if self.threshold_ is None:
            raise ValueError('Threshold is not fitted, call fit() first.')
        elif self.threshold_ > 1 or self.threshold_ < 0:
            raise ValueError('Threshold %.8f is not between 0 and 1.' % (self.threshold_))
        elif np.isnan(self.threshold_):
            logging.getLogger().warn('Threshold is NAN, treating as 0.5')
            threshold = 0.5
        else:
            threshold = self.threshold_
        
        return (S >= threshold).astype(int)
