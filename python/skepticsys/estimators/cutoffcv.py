from .classifiercv import ClassifierCV
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator
from sklearn.utils import check_array
from sklearn.utils.validation import check_is_fitted
from sklearn.metrics import accuracy_score, log_loss
from collections import namedtuple

# parent submodules
import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils import is_pandas, get_proba, get_slice
sys.path.pop(0)
# end parent submodules

class CutoffClassifierCV(ClassifierCV):
    def __init__(self, base_estimator=None, score_func=None, pos_label=1, neg_label=0, cv=3, prefit_callback=None, prefit_params = {}, postfit_callback = None, postfit_params = {}, **kwargs):
        self.score_func = score_func
        self.pos_label = pos_label
        self.neg_label = neg_label
        self.super_kwargs = kwargs
        super().__init__(
            base_estimator=base_estimator
            , cv=cv
            , super_class=_CutoffEstimator
            , super_params={'score_func':self.score_func,'pos_label':self.pos_label,'neg_label':self.neg_label,**self.super_kwargs}
            , prefit_callback=prefit_callback
            , prefit_params=prefit_params
            , postfit_callback=postfit_callback
            , postfit_params=postfit_params
        )

    @property
    def cutoffs(self):
        if self.cutoffs_ is None:
            self.cutoffs_ = [check_cutoff(cl.cutoff_) for cl in self.classifiers_]
        return self.cutoffs_

    @property
    def cutoff(self):
        upper_cutoffs = [unit[0] for unit in self.cutoffs]
        lower_cutoffs = [unit[1] for unit in self.cutoffs]
        return (sum(upper_cutoffs)/len(upper_cutoffs), sum(lower_cutoffs)/len(lower_cutoffs))

    def reset_cv(self):
        super().reset_cv()
        self.cutoffs_ = None

    def predict(self, X):
        check_is_fitted(self, ["classes_", "classifiers_"])
        result = self.classes_[np.argmax(get_proba(self.predict_proba(X)), axis=1)].astype(float)
        result[np.isnan(get_proba(self.predict_proba(X), proba_positive=True))] = np.nan
        return result

class _CutoffEstimator(BaseEstimator):
    """Apply a gray zone cutoff to binary classification probabilities.

    Parameters
    ----------
    decision_threshold: float or callable or transformer, default=0.5
        Decision threshold to evaluate upper and lower cutoffs. If a
        transformer, transformer is fit then the threshold_ property is read.
        If callable, threshold is evaluated on fit().

    kwargs: 
        Extra parameters are passed to the score_func on sample()
    """
    def __init__(self, base_estimator, score_func=None, pos_label=1, neg_label=0, **kwargs):
        self.base_estimator = base_estimator
        self.score_func = score_func
        self.cut_kwargs = kwargs
        self.pos_label = pos_label
        self.neg_label = neg_label
        self.cutoff_ = None
        self.threshold_ = 0.5

    def fit(self, X, y):
        """Find decision_threshold if necessary.

        Parameters
        ----------
        df: array-like, shape (n_samples,n_features) or (n_samples,)
            If `decision_threshold` is a transformer, df is passed to the
            transformer. If `decision_threshold` is a callable, df is the
            predicted scores (y_score).

        y: array-like, shape (n_samples,)
            True classes corresponding to X.

        kwargs:
            Extra parameters are passed to the transformer or callable.

        """
        y_score = self.base_estimator.predict_proba(X)[:, self.pos_label]
        self.cutoff_ = self._get_cutoff(y_score, y, self.score_func, **self.cut_kwargs)

        return self

    def predict(self, X):
        """Apply cutoff to class probabilities and return predicted classes

        Call `fit` first with the true y classes.

        Parameters
        ----------
        y: array-like, shape (n_samples,)
            The predicted class probabilities

        y_classes: array-like, optional
            The predicted classes, if they should be different from `y_true`.
            If `mode` is `class` or `counter`, the replaced classes will still
            be sourced from `y_true`.

        mode: string, optional
            Method to replace cutoff classes, if this should be different
            from instance `mode`. Modes are `nan` (replace with NaN),
            `class` (replace with true class), `counter` (replace with opposite 
            true class), `drop` (remove observations from y)

        Returns
        -------
        self: object
        """
        y_proba = self.predict_proba(X)
        y_pred = np.copy(y_proba)
        y_pred[y_pred <= self.cutoff_[1]] = self.neg_label # lower
        y_pred[y_pred >= self.cutoff_[0]] = self.pos_label # upper
        return y_pred

    def predict_proba(self, X):
        """Transform data by applying the cutoff to class probabilities

        If `mode` is `class` or `counter`, call `fit` first with the true y classes.

        Parameters
        ----------
        y: array-like, shape (n_samples,)
            The predicted class probabilities

        mode: string, optional
            Method to replace cutoff classes, if this should be different
            from instance `mode`. Modes are `nan` (replace with NaN),
            `class` (replace with true class), `counter` (replace with opposite 
            true class), `drop` (remove observations from y)

        Returns
        -------
        y_transformed: array-like, shape (n_samples)
            The transformed class probabilities
        """
        y_score = self.base_estimator.predict_proba(X) #[:, self.pos_label]
        return self.transform(y_score)

    def transform(self, y):
        """Transform data by applying the cutoff to class probabilities

        If `mode` is `class` or `counter`, call `fit` first with the true y classes.

        Parameters
        ----------
        y: array-like, shape (n_samples,)
            The predicted class probabilities

        mode: string, optional
            Method to replace cutoff classes, if this should be different
            from instance `mode`. Modes are `nan` (replace with NaN),
            `class` (replace with true class), `counter` (replace with opposite 
            true class), `drop` (remove observations from y)

        Returns
        -------
        y_transformed: array-like, shape (n_samples)
            The transformed class probabilities
        """
        replace_val = np.nan

        y_proba_cutoff = check_array(np.copy(y))
        #neg_mask, pos_mask = self.get_cutoff_mask(y_proba_cutoff, separate=True)
        mask = self.get_cutoff_mask(get_proba(y_proba_cutoff, proba_positive=True))

        #y_proba_cutoff[np.logical_and(y_proba_cutoff > lower_cutoff, y_proba_cutoff < upper_cutoff)] = replace_val
        #y_proba_cutoff[neg_mask] = replace_val
        #y_proba_cutoff[pos_mask] = replace_val
        y_proba_cutoff[~mask] = replace_val

        return y_proba_cutoff

    def sample(self, df, y):
        return self._sample(df, y)

    def _sample(self, df, y):
        df = check_array(df, ensure_2d=False, copy=True)
        y = check_array(y, ensure_2d=False, copy=True)

        sampled_mask = self.get_cutoff_mask(df)
        df_sampled = df[sampled_mask]
        y_sampled = y[sampled_mask]

        return df_sampled, y_sampled

    def _get_cutoff(self, df, y, score_func, **kwargs):
        # initialize vars
        # if threshold is None:
        #     raise ValueError('decision_threshold must be fit first, call fit()')
        threshold = 0.5

        if score_func is None:
            score_func = _accuracy_count_score
        
        if not callable(score_func):
            raise ValueError('score_func must be callable')

        df = check_array(df, ensure_2d=False, copy=True)
        y = check_array(y, ensure_2d=False, copy=True)

        scoring_name = score_func.__name__
        maximize_score = 'loss' not in scoring_name and 'error' not in scoring_name

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
        
        pred = (df >= threshold).astype(int)
        upper_mask, lower_mask = self.get_decision_mask(df, threshold=threshold)

        # get upper score
        if np.count_nonzero(upper_mask) > 0:
            upper_thresholds = _get_threshold(y[upper_mask], df[upper_mask])
            thr_test = {'threshold': [], 'score': []}
            for thr in upper_thresholds:
                thr_mask = (df >= thr)
                df_thr, y_thr, pred_thr = df[thr_mask], y[thr_mask], pred[thr_mask]
                thr_test['threshold'].append(thr)
                thr_test['score'].append(_call_scorer(score_func, df_thr, y_thr, pred_thr, pos_label=1, **kwargs))

            thr_test_upper = pd.DataFrame(thr_test).sort_values('score', ascending=maximize_score)
                # sort ascending affects thresholds of equal score
                # ascending true gets the more inclusive cutoff, false gets the more exclusive cutoff
            best_idx = thr_test_upper['score'].idxmax() if maximize_score else thr_test_upper['score'].idxmin()
            if not np.isnan(best_idx) and best_idx in thr_test_upper.index:
                upper_score = thr_test_upper['threshold'].loc[best_idx]
            else:
                upper_score = np.nan
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
                thr_test['score'].append(_call_scorer(score_func, df_thr, y_thr, pred_thr, pos_label=0, **kwargs))

            thr_test_lower = pd.DataFrame(thr_test).sort_values('score', ascending=not maximize_score)
                # sort ascending affects thresholds of equal score
                # ascending true gets the more exclusive cutoff, false gets the more inclusive cutoff
            best_idx = thr_test_lower['score'].idxmax() if maximize_score else thr_test_lower['score'].idxmin()
            if not np.isnan(best_idx) and best_idx in thr_test_upper.index:
                lower_score = thr_test_lower['threshold'].loc[best_idx]
            else:
                lower_score = np.nan
        else:
            lower_score = np.nan

        return (upper_score, lower_score)

    def get_decision_mask(self, df, threshold=None):
        if threshold is None:
            if self.threshold_ is None:
                raise ValueError('Decision threshold must be specified or fitted.')
            else:
                threshold = self.threshold_

        if np.isnan(threshold):
            threshold = 0.5

        return ((df >= threshold), (df < threshold)) # upper, lower

    def get_cutoff_mask(self, df, cutoff=None, separate=False):
        if cutoff is None:
            if self.cutoff_ is None:
                raise ValueError('Cutoff tuple must be specified or fitted.')
            else:
                cutoff = self.cutoff_

        if self.threshold_ is None:
            raise ValueError('Decision threshold must be specified or fitted.')
        elif np.isnan(self.threshold_):
            threshold = 0.5
        else:
            threshold = self.threshold_

        upper_cutoff = cutoff[0] if not np.isnan(cutoff[0]) else threshold
        lower_cutoff = cutoff[-1] if not np.isnan(cutoff[-1]) else threshold

        if separate:
            neg_mask = df <= lower_cutoff
            pos_mask = df >= upper_cutoff
            return neg_mask, pos_mask
        else:
            return np.logical_or(df >= upper_cutoff, df <= lower_cutoff)

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

def _accuracy_count_score(y_true, y_pred, pos_label=1, neg_label=0):
    # try precision_score?
    # accuracy_score gives cutoff with least falses possible
    # precision_score allows for some falses, but is prohibitively liberal
    
    acc = accuracy_score(y_true, y_pred)#, pos_label=pos_label)
    cnt = len(y_true[y_true == pos_label])
    counter_acc = accuracy_score(y_true, y_pred)#, pos_label=counter_label)
    counter_cnt = len(y_true[y_true == neg_label])

    return acc*cnt-counter_acc*counter_cnt

def check_cutoff(cutoff):
    result = []
    for unit in cutoff:
        if unit is None:
            raise ValueError('Cutoff is not fitted, call fit() first.')
        elif unit > 1 or unit < 0:
            raise ValueError('Cutoff %.8f is not between 0 and 1.' % (thr))
        elif np.isnan(unit):
            logging.getLogger().warn('Cutoff is NAN, treating as 0.5')
            threshold = 0.5
        else:
            threshold = unit
        result.append(threshold)
    return tuple(result)
