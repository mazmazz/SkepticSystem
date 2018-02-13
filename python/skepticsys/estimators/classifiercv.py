"""Cross-validated claassifier."""

# Author: Marco Z <mar.marcoz@outlook.com>
#         Alexandre Gramfort <alexandre.gramfort@telecom-paristech.fr>
#         Balazs Kegl <balazs.kegl@gmail.com>
#         Jan Hendrik Metzen <jhm@informatik.uni-bremen.de>
#         Mathieu Blondel <mathieu@mblondel.org>
#
# License: BSD 3 clause

import warnings

import numpy as np
import pandas as pd

from sklearn.base import BaseEstimator, ClassifierMixin, clone
from sklearn.preprocessing import LabelBinarizer
from sklearn.utils import check_X_y, check_array, indexable, column_or_1d
from sklearn.utils.validation import check_is_fitted, check_consistent_length
from sklearn.utils.fixes import signature
from sklearn.model_selection import check_cv

import sklearn.metrics as skm

from statistics import mean

# parent submodules
import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils import is_pandas, get_proba, get_slice
sys.path.pop(0)
# end parent submodules

class ClassifierCV(BaseEstimator, ClassifierMixin):
    """Cross-validated classifier.

    With this class, the base_estimator is fit on the train set of each
    cross-validation fold. Then you can either score the original data
    using the in-built attributes or predict new data using the ensemble
    of cross-validated classifiers using predict().

    Parameters
    ----------
    base_estimator : instance BaseEstimator
        The classifier which will be fit on each cross-validated fold. 
        If cv=prefit, the classifier must have been fit already on data.

    cv : integer, cross-validation generator, iterable or "prefit", optional
        Determines the cross-validation splitting strategy.
        Possible inputs for cv are:

        - None, to use the default 3-fold cross-validation,
        - integer, to specify the number of folds.
        - An object to be used as a cross-validation generator.
        - An iterable yielding train/test splits.

        For integer/None inputs, if ``y`` is binary or multiclass,
        :class:`sklearn.model_selection.StratifiedKFold` is used. If ``y`` is
        neither binary nor multiclass, :class:`sklearn.model_selection.KFold`
        is used.

        Refer :ref:`User Guide <cross_validation>` for the various
        cross-validation strategies that can be used here.

        If "prefit" is passed, it is assumed that base_estimator has been
        fitted already and all data is used for calibration.

    super_estimator : instance BaseEstimator
        An estimator to manipulate base_estimator after fitting.

    super_cv : integer, cross-validation generator, iterable or "prefit", optional
        CV generator to pass to super_estimator if there's a "cv" parameter.

    Attributes
    ----------
    classes_ : array, shape (n_classes)
        The class labels.

    classifiers_ : list (len() equal to cv or 1 if cv == "prefit")
        The list of calibrated classifiers, one for each crossvalidation fold,
        which has been fitted on all but the validation fold and calibrated
        on the validation fold.

    split_ : list (len() equal to cv or 1 if cv == "prefit")

    X_ :

    y_ :

    y_pred_cv : list (len() equal to cv or 1 if cv == "prefit")

    y_proba_cv : list (len() equal to cv or 1 if cv == "prefit")

    y_true_cv : list (len() equal to cv or 1 if cv == "prefit")

    y_pred :

    y_proba :

    y_true :


    """
    def __init__(self, base_estimator=None, cv=3, super_class=None, super_params={}, prefit_callback=None, prefit_params = {}, postfit_callback = None, postfit_params = {}):
        self.base_estimator = base_estimator
        self.cv = cv
        self.super_class = super_class
        self.super_params = super_params
        self.prefit_callback = prefit_callback
        self.prefit_params = prefit_params
        self.postfit_callback = postfit_callback
        self.postfit_params = postfit_params
        self.reset_cv()

    def fit(self, X, y, sample_weight=None, **kwargs):
        """Fit the base estimator

        Parameters
        ----------
        X : array-like, shape (n_samples, n_features)
            Training data.

        y : array-like, shape (n_samples,)
            Target values.

        sample_weight : array-like, shape = [n_samples] or None
            Sample weights. If None, then samples are equally weighted.

        kwargs :
            Extra args are passed to the base classifier for fitting.

        Returns
        -------
        self : object
            Returns an instance of self.
        """
        base_X, base_y = X, y
        #X, y = check_X_y(X, y, accept_sparse=['csc', 'csr', 'coo'],
        #                 force_all_finite=False)
        #X, y = indexable(X, y)
        le = LabelBinarizer().fit(y)
        self.classes_ = le.classes_
        
        # Check that each cross-validation fold can have at least one
        # example per class
        n_folds = self.cv if isinstance(self.cv, int) \
            else self.cv.n_folds if hasattr(self.cv, "n_folds") else None
        if n_folds and \
                np.any([np.sum(y == class_) < n_folds for class_ in
                        self.classes_]):
            raise ValueError("Requesting %d-fold cross-validation but provided"
                             " less than %d examples for at least one class."
                             % (n_folds, n_folds))

        classifiers = []
        base_estimator = self.base_estimator
        super_class = self.super_class

        if self.cv == "prefit":
            if super_class is not None:
                super_estimator = super_class(base_estimator, **self.super_params)
                super_fit_parameters = signature(super_estimator.fit).parameters
                if sample_weight is not None and 'sample_weight' in super_fit_parameters:
                    super_estimator.fit(X, y, sample_weight)
                else:
                    super_estimator.fit(X, y)
                classifiers.append(super_estimator)
            else:
                classifiers.append(base_estimator)
            splits = [list(range(len(X))), []]
        else:
            cv = check_cv(self.cv, y, classifier=True)
            fit_parameters = signature(base_estimator.fit).parameters
            estimator_name = type(base_estimator).__name__
            if (sample_weight is not None
                    and "sample_weight" not in fit_parameters):
                warnings.warn("%s does not support sample_weight. Samples"
                              " weights are only used for the calibration"
                              " itself." % estimator_name)
                base_estimator_sample_weight = None
            else:
                if sample_weight is not None:
                    sample_weight = check_array(sample_weight, ensure_2d=False)
                    check_consistent_length(y, sample_weight)
                base_estimator_sample_weight = sample_weight
            for k in kwargs:
                if k not in fit_parameters:
                    warnings.warn('%s does not support %s, dropping.'
                                    % (estimator_name, k))
                    kwargs.pop(k)
            splits = list(cv.split(X, y))
            for i, (train, test) in enumerate(splits):
                this_estimator = clone(base_estimator)
                X_train, y_train, X_test, y_test = get_slice(X, rows=train), get_slice(y, rows=train), get_slice(X, rows=test), get_slice(y, rows=test)

                if callable(self.prefit_callback):
                    if self.prefit_callback(X_train, y_train, X_test, y_test, i, **self.prefit_params) is False:
                        raise ValueError('Fitting aborted by prefit_params')

                if base_estimator_sample_weight is not None:
                    this_estimator.fit(
                        X_train, y_train,
                        sample_weight=base_estimator_sample_weight[train],
                        **kwargs)
                else:
                    this_estimator.fit(X_train, y_train, **kwargs)

                if super_class is not None:
                    super_params = {**self.super_params}
                    if 'classes' in signature(super_class.__init__).parameters:
                        super_params['classes'] = self.classes_
                    super_estimator = super_class(this_estimator, **super_params)
                    
                    super_fit_parameters = signature(super_estimator.fit).parameters
                    if sample_weight is not None and 'sample_weight' in super_fit_parameters:
                        super_estimator.fit(X_test, y_test, sample_weight)
                    else:
                        super_estimator.fit(X_test, y_test)

                    if callable(self.postfit_callback):
                        if self.postfit_callback(X_train, y_train, X_test, y_test, i, **self.postfit_params) is False:
                            raise ValueError('Fitting aborted by postfit_params')
                    classifiers.append(super_estimator)
                else:
                    if callable(self.postfit_callback):
                        if self.postfit_callback(X_train, y_train, X_test, y_test, i, **self.postfit_params) is False:
                            raise ValueError('Fitting aborted by postfit_params')

                    classifiers.append(this_estimator)

        self.reset_cv()
        self.X_, self.y_ = base_X, base_y
        self.classifiers_ = classifiers
        self.split_ = splits

        return self

    @property
    def y_pred_cv(self):
        if self.y_pred_cv_ is None:
            self.y_pred_cv_ = []
            for i, (train, test) in enumerate(self.split_):
                if hasattr(self.classifiers_[i], 'predict') and callable(self.classifiers_[i].predict):
                    result = self.classifiers_[i].predict(get_slice(self.X_, rows=test))
                    if is_pandas(self.y_):
                        result = pd.Series(result, index=self.y_true_cv[i].index)
                    self.y_pred_cv_.append(result)
                elif hasattr(self.classifiers_[i], 'predict_proba') and callable(self.classifiers_[i].predict_proba):
                    proba = self.classifiers_[i].predict_proba(get_slice(self.X_, rows=test))
                    result = self.classes_[np.argmax(proba, axis=1)]
                    if is_pandas(self.y_):
                        result = pd.Series(result, index=self.y_true_cv[i].index)
                    self.y_pred_cv_.append(result)
                else:
                    raise ValueError('Base classifier does not have "predict" or "predict_proba" callable.')
        return self.y_pred_cv_

    @property
    def y_proba_cv(self):
        if self.y_proba_cv_ is None:
            self.y_proba_cv_ = []
            for i, (train, test) in enumerate(self.split_):
                if hasattr(self.classifiers_[i], 'predict_proba') and callable(self.classifiers_[i].predict_proba):
                    result = self.classifiers_[i].predict_proba(get_slice(self.X_, rows=test))
                    if is_pandas(self.y_):
                        result = pd.DataFrame(result, index=self.y_true_cv[i].index)
                    self.y_proba_cv_.append(result)
                else:
                    raise ValueError('Base classifier does not have "predict_proba" callable.')
        return self.y_proba_cv_

    @property
    def y_true_cv(self):
        if self.y_true_cv_ is None:
            self.y_true_cv_ = []
            for train, test in self.split_:
                self.y_true_cv_.append(get_slice(self.y_, rows=test))
        return self.y_true_cv_

    @property
    def y_pred(self):
        if self.y_pred_ is None:
            if is_pandas(self.y_):
                self.y_pred_ = pd.concat(self.y_pred_cv)
            else:
                self.y_pred_ = np.concatenate(self.y_pred_cv)
        return self.y_pred_

    @property
    def y_proba(self):
        if self.y_proba_ is None:
            if is_pandas(self.y_):
                self.y_proba_ = pd.concat(self.y_proba_cv)
            else:
                self.y_proba_ = np.concatenate(self.y_proba_cv)
        return self.y_proba_

    @property
    def y_true(self):
        if self.y_true_ is None:
            if is_pandas(self.y_):
                self.y_true_ = pd.concat(self.y_true_cv)
            else:
                self.y_true_ = np.concatenate(self.y_true_cv)
        return self.y_true_

    def reset_cv(self):
        self.classifiers_, self.split_, self.X_, self.y_ = None, None, None, None
        self.y_pred_cv_, self.y_proba_cv_, self.y_true_cv_, self.y_pred_, self.y_proba_, self.y_true_ = None, None, None, None, None, None

    def score_cv(self, scorer=skm.accuracy_score, aggregate='average', proba_positive=False, **kwargs):
        """Score fitted data.

        Parameters
        ----------
        scorer : callable
            Callable that returns one single score. Its parameters must have
            y_true, y_truth, or y; and either y_pred; or y_proba or y_df.

        aggregate : 'average', 'concatenate', 'cv', or 'full'
            The method to use for scoring the test sets. 'average' gives a mean
            average of each test set score; 'concatenate' combines each
            test set's prediction into a sequential list, then scores that
            combined list; 'cv' scores each split individually and returns
            a list of scores per split; 'full' returns all outputs.

        proba_positive : boolean, default False
            Pass only the positive class for y_proba

        kwargs : optional
            Additional parameters are passed to the scorer callable.
        """
        score_args = kwargs
        score_parameters = signature(scorer).parameters

        # todo: y_true, y_pred, y_proba dropping with NaN
        
        if aggregate in ['concatenate','concat','full','all']:
            # drop nan from truth
            proba, pred = self.y_proba, self.y_pred
            nan_mask = np.any(np.stack([np.isnan(get_proba(proba, proba_positive=True)), np.isnan(pred)],axis=1),axis=1)
            proba, pred, truth = proba[~nan_mask], pred[~nan_mask], self.y_true[~nan_mask]

            if 'y_true' in score_parameters:
                score_args['y_true'] = truth
            elif 'y_truth' in score_parameters:
                score_args['y_truth'] = truth
            elif 'y' in score_parameters:
                score_args['y'] = truth

            if 'y_pred' in score_parameters:
                score_args['y_pred'] = pred

            if 'y_proba' in score_parameters:
                score_args['y_proba'] = get_proba(proba, proba_positive=proba_positive)
            elif 'y_prob' in score_parameters:
                score_args['y_prob'] = get_proba(proba, proba_positive=proba_positive)
            elif 'y_df' in score_parameters:
                score_args['y_df'] = get_proba(proba, proba_positive=proba_positive)
            
            if aggregate not in ['full','all']:
                return scorer(**score_args)
            else:
                concat_result = scorer(**score_args)
        
        if aggregate not in ['concatenate','concat']:
            scores = []
            for pred, proba, truth in zip(self.y_pred_cv, self.y_proba_cv, self.y_true_cv):
                # drop nan from truth
                nan_mask = np.any(np.stack([np.isnan(get_proba(proba, proba_positive=True)), np.isnan(pred)],axis=1),axis=1)
                proba, pred, truth = proba[~nan_mask], pred[~nan_mask], truth[~nan_mask]

                if 'y_true' in score_parameters:
                    score_args['y_true'] = truth
                elif 'y_truth' in score_parameters:
                    score_args['y_truth'] = truth
                elif 'y' in score_parameters:
                    score_args['y'] = truth

                if 'y_pred' in score_parameters:
                    score_args['y_pred'] = pred

                if 'y_proba' in score_parameters:
                    score_args['y_proba'] = get_proba(proba, proba_positive=proba_positive)
                elif 'y_prob' in score_parameters:
                    score_args['y_prob'] = get_proba(proba, proba_positive=proba_positive)
                elif 'y_df' in score_parameters:
                    score_args['y_df'] = get_proba(proba, proba_positive=proba_positive)
                scores.append(scorer(**score_args))
            
            if aggregate in ['average','avg']:
                return mean(scores)
            elif aggregate in ['cv','crossval','split']:
                return scores
            else:
                avg_result = mean(scores)
                cv_result = scores
        
        return {
            'concat': concat_result
            , 'avg': avg_result
            , 'cv': cv_result
        }

    def predict_proba(self, X):
        """Posterior probabilities of classification

        This function returns posterior probabilities of classification
        according to each class on an array of test vectors X.

        Parameters
        ----------
        X : array-like, shape (n_samples, n_features)
            The samples.

        Returns
        -------
        C : array, shape (n_samples, n_classes)
            The predicted probas.
        """

        check_is_fitted(self, ["classes_", "classifiers_"])
        # X = check_array(X, accept_sparse=['csc', 'csr', 'coo'],
        #                 force_all_finite=False)
        # Compute the arithmetic mean of the predictions of the classifiers
        mean_proba = np.zeros((X.shape[0], len(self.classes_)))
        for classifier in self.classifiers_:
            proba = classifier.predict_proba(X)
            mean_proba += proba

        mean_proba /= len(self.classifiers_)

        return mean_proba

    def predict(self, X):
        """Predict the target of new samples. Can be different from the
        prediction of the uncalibrated classifier.

        Parameters
        ----------
        X : array-like, shape (n_samples, n_features)
            The samples.

        Returns
        -------
        C : array, shape (n_samples,)
            The predicted class.
        """
        check_is_fitted(self, ["classes_", "classifiers_"])
        return self.classes_[np.argmax(self.predict_proba(X), axis=1)]
