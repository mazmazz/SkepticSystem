import numpy as np
import pandas as pd
import sklearn.pipeline
from sklearn.pipeline import _name_estimators, _transform_one, _fit_transform_one
from sklearn.externals.joblib import Parallel, delayed
from scipy import sparse

class FeatureUnion(sklearn.pipeline.FeatureUnion):
    """Concatenates results of multiple transformer objects. Pandas-aware.

    This estimator applies a list of transformer objects in parallel to the
    input data, then concatenates the results. This is useful to combine
    several feature extraction mechanisms into a single transformer.

    Parameters of the transformers may be set using its name and the parameter
    name separated by a '__'. A transformer may be replaced entirely by
    setting the parameter with its name to another transformer,
    or removed by setting to ``None``.

    Read more in the :ref:`User Guide <feature_union>`.

    Parameters
    ----------
    transformer_list : list of (string, transformer) tuples
        List of transformer objects to be applied to the data. The first
        half of each tuple is the name of the transformer.

    n_jobs : int, optional
        Number of jobs to run in parallel (default 1).

    transformer_weights : dict, optional
        Multiplicative weights for features per transformer.
        Keys are transformer names, values the weights.

    output : str
        Concatenated output is 'pandas' or 'numpy'. Set to 'auto' to determine
        automatically from inputted data (default 'auto').
    """
    def __init__(self, transformer_list, n_jobs=1, transformer_weights=None, output='auto'):
        self.output = output
        super().__init__(transformer_list, n_jobs=n_jobs, transformer_weights=transformer_weights)

    def fit_transform(self, X, y=None, **fit_params):
        """Fit all transformers, transform the data and concatenate results.

        Parameters
        ----------
        X : iterable or array-like, depending on transformers
            Input data to be transformed.

        y : array-like, shape (n_samples, ...), optional
            Targets for supervised learning.

        Returns
        -------
        X_t : array-like or sparse matrix, shape (n_samples, sum_n_components)
            hstack of results of transformers. sum_n_components is the
            sum of n_components (output dimension) over transformers.
        """
        if self.output.lower() not in ['pandas','numpy','pd','np']:
            output = ('pandas' if isinstance(X, pd.DataFrame) or isinstance(X, pd.Series)
                     else 'numpy')
        else:
            output = self.output
        output_pandas = output.lower() in ['pandas','pd']

        self._validate_transformers()
        result = Parallel(n_jobs=self.n_jobs)(
            delayed(_fit_transform_one)(trans, weight, X, y,
                                        **fit_params)
            for name, trans, weight in self._iter())

        if not result:
            # All transformers are None
            if output_pandas:
                return pd.DataFrame(0, index=X.index if isinstance(X, pd.DataFrame) or isinstance(X, pd.Series) 
                                                     else range(X.shape[0])
                                     , columns=range(0))
            else:
                return np.zeros((X.shape[0], 0))
        Xs, transformers = zip(*result)
        self._update_transformer_list(transformers)
        if any(sparse.issparse(f) for f in Xs):
            if output_pandas:
                Xs = pd.concat(Xs, axis=1)
            else:
                Xs = sparse.hstack(Xs).tocsr()
        else:
            if output_pandas:
                Xs = pd.concat(Xs, axis=1)
            else:
                Xs = np.hstack(Xs)
        return Xs

    def transform(self, X):
        """Transform X separately by each transformer, concatenate results.

        Parameters
        ----------
        X : iterable or array-like, depending on transformers
            Input data to be transformed.

        Returns
        -------
        X_t : array-like or sparse matrix, shape (n_samples, sum_n_components)
            hstack of results of transformers. sum_n_components is the
            sum of n_components (output dimension) over transformers.
        """
        if self.output.lower() not in ['pandas','numpy','pd','np']:
            output = ('pandas' if isinstance(X, pd.DataFrame) or isinstance(X, pd.Series)
                     else 'numpy')
        else:
            output = self.output
        output_pandas = output.lower() in ['pandas','pd']

        Xs = Parallel(n_jobs=self.n_jobs)(
            delayed(_transform_one)(trans, weight, X)
            for name, trans, weight in self._iter())
        if not Xs:
            # All transformers are None
            if output_pandas:
                return pd.DataFrame(0, index=X.index if isinstance(X, pd.DataFrame) or isinstance(X, pd.Series) 
                                                     else range(X.shape[0])
                                     , columns=range(0))
            else:
                return np.zeros((X.shape[0], 0))
        if any(sparse.issparse(f) for f in Xs):
            if output_pandas:
                Xs = pd.concat(Xs, axis=1)
            else:
                Xs = sparse.hstack(Xs).tocsr()
        else:
            if output_pandas:
                Xs = pd.concat(Xs, axis=1)
            else:
                Xs = np.hstack(Xs)
        return Xs

def make_union(*transformers, **kwargs):
    """Construct a FeatureUnion from the given transformers.

    This is a shorthand for the FeatureUnion constructor; it does not require,
    and does not permit, naming the transformers. Instead, they will be given
    names automatically based on their types. It also does not allow weighting.

    Parameters
    ----------
    *transformers : list of estimators

    n_jobs : int, optional
        Number of jobs to run in parallel (default 1).

    Returns
    -------
    f : FeatureUnion

    Examples
    --------
    >>> from sklearn.decomposition import PCA, TruncatedSVD
    >>> from sklearn.pipeline import make_union
    >>> make_union(PCA(), TruncatedSVD())    # doctest: +NORMALIZE_WHITESPACE
    FeatureUnion(n_jobs=1,
           transformer_list=[('pca',
                              PCA(copy=True, iterated_power='auto',
                                  n_components=None, random_state=None,
                                  svd_solver='auto', tol=0.0, whiten=False)),
                             ('truncatedsvd',
                              TruncatedSVD(algorithm='randomized',
                              n_components=2, n_iter=5,
                              random_state=None, tol=0.0))],
           transformer_weights=None)
    """
    n_jobs = kwargs.pop('n_jobs', 1)
    if kwargs:
        # We do not currently support `transformer_weights` as we may want to
        # change its type spec in make_union
        raise TypeError('Unknown keyword arguments: "{}"'
                        .format(list(kwargs.keys())[0]))
    return FeatureUnion(_name_estimators(transformers), n_jobs=n_jobs)
