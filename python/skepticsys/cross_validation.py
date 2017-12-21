from sklearn.model_selection import BaseCrossValidator
from sklearn.utils.validation import (_num_samples, indexable)
from math import floor
import numpy as np

def _get_size(n_samples, size, neg_mode='pass'):
    if size is not None:
        if abs(size) < 1: 
            size = floor(abs(n_samples*size)) * (-1 if size < 0 else 1) # floor towards abs value, not lowest number

        if size < 0:
            if neg_mode == 'subtract':
                return n_samples+size
            else: # neg_mode == 'pass':
                return size
    
    return size

class SingleSplit(BaseCrossValidator):
    """Single split cross-validator.

    Provides train/test indices to split time series data samples
    that are observed at fixed time intervals, in train/test sets.
    In each split, test indices must be higher than before, and thus shuffling
    in cross validator is inappropriate.

    Parameters
    ----------
    test_size: int or float, default=0.1
        Size of test set. If this is a fraction between 0 and 1, size is figured
        from the proportion of sample size, dependent on `initial_index`

    initial_index: int or float, default=0
        Initial index of train series. This may determine both train and test
        size, if `test_size` is a fraction. If `initial_index` is a fraction
        between 0 and 1, size is figured from the proportion of total sample
        size.
    """
    def __init__(self, test_size=0.1, initial_index=0):
        self.test_size = test_size
        self.initial_index = initial_index
    
    def split(self, X, y=None, groups=None):
        """Generate indices to split data into training and test set.

        Parameters
        ----------
        X : array-like, shape (n_samples, n_features)
            Training data, where n_samples is the number of samples
            and n_features is the number of features.

        y : array-like, shape (n_samples,)
            Always ignored, exists for compatibility.

        groups : array-like, with shape (n_samples,), optional
            Always ignored, exists for compatibility.

        Returns
        -------
        train : ndarray
            The training set indices for that split.

        test : ndarray
            The testing set indices for that split.
        """
        X, y, groups = indexable(X, y, groups)
        n_samples = _num_samples(X)
        initial_index = max(0, _get_size(n_samples, self.initial_index, neg_mode='subtract'))
        final_index = n_samples
        test_size = _get_size(n_samples-initial_index, self.test_size, neg_mode='subtract')
        
        indices = np.arange(n_samples)
        train = indices[initial_index:-test_size]
        test  = indices[-test_size:final_index]

        yield train, test

    def get_n_splits(self, X=None, y=None, groups=None):
        # I've only seen this for test suites, so pass a fake number
        return 2

class WindowSplit(BaseCrossValidator):
    """Expanding or sliding window cross-validator

    Provides train/test indices to split time series data samples
    that are observed at fixed time intervals, in train/test sets.
    In each split, test indices must be higher than before, and thus shuffling
    in cross validator is inappropriate.

    Note that unlike standard cross-validation methods, successive
    training sets are supersets of those that come before them.

    Read more in the :ref:`User Guide <cross_validation>`.

    Parameters
    ----------
    test_size : int, default=1
        Size of test set. If this is a fraction between 0 and 1, size is figured
        from the proportion of sample size. If this is set to `None`, test set
        always expands to end of samples, but only `step_size` is added to each
        successive train set.

    step_size : int, default=1
        Number of samples to step at each split. If this is a fraction between 0
        and 1, size is figured from the proportion of sample size.

    delay_size : int, default=0
        Number of samples to skip between train set and test set. If this is 
        a fraction between 0 and 1, size is figured from the proportion of 
        sample size.

    sliding_size : int, optional
        Size of sliding train set. If this is a fraction between 0 and 1, size
        is figured from the proportion of sample size. If this is set to `None`,
        train set will be expanding.

    initial_index : int, default=0
        Index of the first sample for the initial test set. If this is a fraction 
        between 0 and 1 or 0 and -1, index is found at the proportion of sample size.
        If this is a negative number, subtract the amount from sample size.

    final_index: int, optional
        Last index to include in a split. Following indexes are
        ignored and not put into a set. If this is a fraction between 0 and 1
        or 0 and -1, index is found at the proportion of sample size. If this
        is a negative number, subtract the amount from sample size. If this
        is ommitted, include all indexes in the window sets.

    min_test_size : int, optional
        Smallest test set size allowed, for expanding test sets (`test_size = None`).
        If `test_size` is not `None`, this is ignored. If this is a fraction between
        0 and 1, size is figured from the proportion of sample size.

    force_sliding_min: bool, default=True
        If `sliding_size` is set, force the first train set to be of `sliding_size`.
        This overrides `initial_index` if `initial_index` is less than `sliding_size`.

    test_remainder : bool, default=False
        Allow last split if test set size is smaller than `test_size` and at least 1.
    """
    def __init__(self, test_size=1, step_size=1, delay_size=0, sliding_size=None, initial_index=0, final_index=None, min_test_size=None, force_sliding_min=True, test_remainder=False):
        self.test_size = test_size
        self.step_size = step_size
        self.delay_size = delay_size
        self.sliding_size = sliding_size
        self.initial_index = initial_index
        self.final_index = final_index
        self.min_test_size = min_test_size
        self.force_sliding_min = force_sliding_min
        self.test_remainder = test_remainder

    def split(self, X, y=None, groups=None):
        """Generate indices to split data into training and test set.

        Parameters
        ----------
        X : array-like, shape (n_samples, n_features)
            Training data, where n_samples is the number of samples
            and n_features is the number of features.

        y : array-like, shape (n_samples,)
            Always ignored, exists for compatibility.

        groups : array-like, with shape (n_samples,), optional
            Always ignored, exists for compatibility.

        Returns
        -------
        train : ndarray
            The training set indices for that split.

        test : ndarray
            The testing set indices for that split.

        Notes
        -----
        Randomized CV splitters may return different results for each call of
        split. You can make the results identical by setting ``random_state``
        to an integer.
        """

        X, y, groups = indexable(X, y, groups)
        n_samples = _num_samples(X)
        test_size = _get_size(n_samples, self.test_size)
        step_size = _get_size(n_samples, self.step_size)
        delay_size = _get_size(n_samples, self.delay_size)
        sliding_size = _get_size(n_samples, self.sliding_size)
        initial_index = _get_size(n_samples, self.initial_index, neg_mode='subtract')
        final_index = _get_size(n_samples, self.final_index, neg_mode='subtract')
        min_test_size = _get_size(n_samples, self.min_test_size)
        test_remainder = self.test_remainder
        force_sliding_min = self.force_sliding_min
        test_expanding = test_size is None
        self._do_validation(n_samples, step_size, initial_index)

        if not final_index is None and final_index < n_samples:
            n_samples = final_index

        indices = np.arange(n_samples)
        if not sliding_size is None and force_sliding_min:
            i = max(sliding_size-1, initial_index-1)
        else: # sliding window
            i = max(0, initial_index-1)
        remainder_run = False
        while True:
            if sliding_size is None: # expanding window
                train_start, train_end = 0, i+1
                test_start , test_end  = i+1+delay_size, n_samples if test_expanding else (i+1+delay_size)+test_size
            else: # sliding window
                train_start, train_end = max(0, i-sliding_size+1), i+1
                test_start , test_end  = i+1+delay_size, n_samples if test_expanding else (i+1+delay_size)+test_size
            
            if (not test_remainder and test_end > n_samples) \
                or test_start >= n_samples \
                or test_end-test_start <= 0 \
                or (test_expanding and not min_test_size is None and test_end-test_start < min_test_size):
                break

            train = indices[train_start:train_end]
            test  = indices[test_start:min(test_end, n_samples)]

            yield train, test
            
            next_end = train_end+delay_size+step_size if test_expanding else test_end+step_size
            if next_end > n_samples:
                if test_remainder and not remainder_run: 
                    remainder_run = True
                    i += step_size
                else:
                    break
            else:
                i += step_size

    def get_n_splits(self, X=None, y=None, groups=None):
        # I've only seen this for test suites, so pass a fake number
        return 2

    def _do_validation(self, n_samples, step_size, initial_index):
        if step_size < 1:
            raise ValueError('step_size must be greater than zero.')
            
        if initial_index < 0:
            raise ValueError('initial_index must be positive.')
            