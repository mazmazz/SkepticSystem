import numpy as np
import pandas as pd
from imblearn.base import SamplerMixin
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.utils import check_array#, check_X_y
from collections import namedtuple
from sklearn.metrics import accuracy_score #, precision_score

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

class CutoffSampler(SamplerMixin):
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
    UpperLower = namedtuple('UpperLower', ['upper','lower'])
    def __init__(self, score_func=None, decision_threshold=0.5, dec_kwargs=None, cut_kwargs=None):
        self.score_func = score_func
        self.dec_kwargs = dec_kwargs if dec_kwargs is not None else {}
        self.cut_kwargs = cut_kwargs if cut_kwargs is not None else {}
        self.decision_threshold = decision_threshold
        self.threshold_ = decision_threshold if not isinstance(decision_threshold, BaseEstimator) and not callable(decision_threshold) else None
        self.cutoff_ = None

    def fit(self, df, y):
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
        self.threshold_ = self._get_decision_threshold(df, y, self.decision_threshold, **self.dec_kwargs)
        self.cutoff_ = self._get_cutoff(df, y, self.score_func, self.threshold_, **self.cut_kwargs)

        return self

    def _get_decision_threshold(self, df, y, decision_input, **kwargs):
        if isinstance(decision_input, BaseEstimator):
            decision_input.fit(df, y, **kwargs)
            threshold = decision_input.threshold_
        elif callable(decision_input):
            df = check_array(df, ensure_2d=False, copy=True)
            y = check_array(y, ensure_2d=False, copy=True)
            threshold = decision_input(df, y, **kwargs)
        else:
            threshold = decision_input

        return threshold

    def _get_cutoff(self, df, y, score_func, threshold, **kwargs):
        # initialize vars
        if threshold is None:
            raise ValueError('decision_threshold must be fit first, call fit()')

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

        return self.UpperLower(upper=(df >= threshold), lower=(df < threshold))

    def get_cutoff_mask(self, df, cutoff=None):
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
            threshold = 0.5

        upper_cutoff = cutoff[0] if not np.isnan(cutoff[0]) else threshold
        lower_cutoff = cutoff[-1] if not np.isnan(cutoff[-1]) else threshold

        return np.logical_or(df >= upper_cutoff, df <= lower_cutoff)

    def sample(self, df, y):
        return self._sample(df, y)

    def _sample(self, df, y):
        df = check_array(df, ensure_2d=False, copy=True)
        y = check_array(y, ensure_2d=False, copy=True)

        sampled_mask = self.get_cutoff_mask(df)
        df_sampled = df[sampled_mask]
        y_sampled = y[sampled_mask]

        return df_sampled, y_sampled

class CutoffTransformer(BaseEstimator, TransformerMixin):
    def __init__(self, cutoff, upper_cutoff=None, y_true=None, mode='nan'):
        """Create a CutoffTransformer object.

        Parameters
        ----------
        cutoff: int
            The score cutoff to use for both classes, inclusive.
        upper_cutoff: int, optional
            The score cutoff to use for the positive class, if it should be different
            from `cutoff`. Inclusive.
        y_true: array-like, optional
            True classes for replacement using `class` and `counter`. This can be
            changed using `fit`.
        mode: float or string, default='nan'
            Method to replace cutoff probabilities. Modes are `nan` (replace with NaN),
            `class` (replace with 100% probability for test class), `counter` (replace
            with 0% probability for test class), `drop` (remove observations from y)
        """
        self.cutoff = cutoff
        self.upper_cutoff = upper_cutoff
        self.mode = mode
        self.y_true = y_true
    
    def fit(self, y):
        """Fit `CutoffTransformer` with the true y classes.

        Parameters
        ----------
        y: array-like, shape (n_samples,)
            The predicted class probabilities

        Returns
        -------
        self: object
        """
        self.y_true = y
        return self

    def predict(self, y, y_classes=None, mode=None):
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
        mask = np.logical_or(*self._get_mask(y))
        if mode is None: mode = self.mode
        if y_classes is None: y_classes = self.y_true

        y_cutoff = np.copy(y_classes)
        if mode == 'class':
            y_cutoff[mask] = self.y_true[mask]
        elif mode == 'counter':
            y_cutoff[mask] = abs(self.y_true[mask]-1)
        elif mode == 'drop':
            y_cutoff = y_cutoff[~mask]
        else:
            y_cutoff = y_cutoff.astype(np.float64, copy=False)
            y_cutoff[mask] = np.nan

        return y_cutoff

    def predict_proba(self, y, mode=None):
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
        return self.transform(y, mode=mode)

    def transform(self, y, mode=None):
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
        if mode is None: mode = self.mode
        if not (mode in ['class','counter'] and not self.y_true is None) and not mode in ['drop']:
            if self.mode == 'nan' or self.y_true is None:
                replace_val = np.nan
            else:
                replace_val = self.mode
            mode = 'val'

        y_proba_cutoff = check_array(np.copy(y))
        neg_mask, pos_mask = self._get_mask(y_proba_cutoff)

        if mode == 'class':
            #y_proba_cutoff[np.logical_and(y_proba_cutoff > lower_cutoff, y_proba_cutoff < upper_cutoff)] = self.y_true[np.logical_and(y_proba_cutoff > lower_cutoff, y_proba_cutoff < upper_cutoff)]
            y_proba_cutoff[neg_mask, 0] = abs(self.y_true[neg_mask]-1)
            y_proba_cutoff[neg_mask, 1] = self.y_true[neg_mask]
            y_proba_cutoff[pos_mask, 0] = self.y_true[pos_mask]
            y_proba_cutoff[pos_mask, 1] = abs(self.y_true[pos_mask]-1)
        elif mode == 'counter':
            #y_proba_cutoff[np.logical_and(y_proba_cutoff > lower_cutoff, y_proba_cutoff < upper_cutoff)] = abs(self.y_true[np.logical_and(y_proba_cutoff > lower_cutoff, y_proba_cutoff < upper_cutoff)]-1)
            y_proba_cutoff[neg_mask, 0] = self.y_true[neg_mask]
            y_proba_cutoff[neg_mask, 1] = abs(self.y_true[neg_mask]-1)
            y_proba_cutoff[pos_mask, 0] = abs(self.y_true[pos_mask]-1)
            y_proba_cutoff[pos_mask, 1] = self.y_true[pos_mask]
        elif mode == 'drop':
            y_proba_cutoff = y_proba_cutoff[~np.logical_or(neg_mask, pos_mask)]
        else:
            #y_proba_cutoff[np.logical_and(y_proba_cutoff > lower_cutoff, y_proba_cutoff < upper_cutoff)] = replace_val
            y_proba_cutoff[neg_mask] = replace_val
            y_proba_cutoff[pos_mask] = replace_val

        return y_proba_cutoff

    def _get_mask(self, y):
        """Get boolean mask for class probabilities that fall outside the cutoff.

        Parameters
        ----------
        y: array-like, shape (n_samples,)
            The predicted class probabilities

        Returns
        -------
        neg_mask: array-like, shape(n_samples,)
            Mask for negative class probabilities outside the cutoff.
        pos_mask: array-like, shape(n_samples,)
            Mask for positive class probabilities outside the cutoff.
        """
        lower_cutoff = 1-self.cutoff
        upper_cutoff = 1-self.upper_cutoff if not self.upper_cutoff is None else 1-self.cutoff

        neg_mask = np.logical_and(y[:,0] >= y[:,1]
                                        , y[:,0] < lower_cutoff)
        pos_mask = np.logical_and(y[:,0] < y[:,1]
                                        , y[:,1] < upper_cutoff)

        return neg_mask, pos_mask

class CutoffEstimator(CutoffTransformer):
    """Transformer for using cutoff thresholds for class probabilities.

    Parameters
    ----------
    estimator : object
        The base estimator from which the transformer is built.
    """

    def __init__(self, estimator, cutoff, upper_cutoff=None, mode='nan', y_true=None):
        """Create a CutoffEstimator object.

        Parameters
        ----------
        estimator: object with fit, predict, and predict_proba methods.
            The estimator to use.
        cutoff: int
            The score cutoff to use for both classes, inclusive.
        upper_cutoff: int, optional
            The score cutoff to use for the positive class, if it should be different
            from `cutoff`. Inclusive.
        mode: float or string, default='nan'
            The value to replace rejected class probabilities, or the preset mode to
            use. Modes are `nan` (replace with NaN), `class` (replace with 100% probability
            for test class), `counter` (replace with 0% probability for test class)
        y_true: array-like, optional
            True classes for replacement using `class` and `counter`. This can be
            changed using `fit`.
        """
        self.estimator = estimator
        self.cutoff = cutoff
        self.upper_cutoff = upper_cutoff
        self.mode = mode
        self.y_true = y_true

    def fit(self, X, y=None, **fit_params):
        """Fit the CutoffEstimator meta-transformer.

        Parameters
        ----------
        X: array-like of shape (n_samples, n_features)
            The training input samples. If `y` is None, this is interpreted as y_true for CutoffTransformer.fit.
        y: array-like, shape (n_samples,)
            The target values (integers that correspond to classes in classification, real numbers in regression).
        fit_params:
            Other estimator-specific parameters.

        Returns
        -------
        self: object
            Returns a copy of the estimator
        """
        if y is None: # this is a hack to call CutoffTransformer.fit() with y_true
            super().fit(X) 
        else:
            self.estimator.fit(X, y, **fit_params)
        return self

    def predict(self, X):
        y_pred = self.estimator.predict(X)
        y_proba = self.estimator.predict_proba(X)
        return super().predict(y_proba, y_classes=y_pred)

    def predict_proba(self, X):
        y_proba = self.estimator.predict_proba(X)
        return super().predict_proba(y_proba)

    def transform(self, y, mode=None):
        """Transform data by applying the cutoff to class probabilities

        Call `fit` first with the true y classes.

        Parameters
        ----------
        y: array-like, shape (n_samples,)
            The predicted class probabilities

        Returns
        -------
        y_transformed: array-like, shape (n_samples)
            The transformed class probabilities
        """
        return super().transform(y, mode=mode)

    def score(self, X, y=None, sample_weight=None):
        y_pred = self.predict(X)
        if len(y) != len(y_pred):
            raise ValueError('y and y_pred are different lengths.{}'.format(' When mode=drop, y must not have observations that were dropped from y_pred. Use mode=nan instead.' if self.mode == 'drop' else ''))

        y_true = np.copy(y)[~np.isnan(y_pred)]
        y_pred = y_pred[~np.isnan(y_pred)]

        return accuracy_score(y_true, y_pred, sample_weight=sample_weight)
