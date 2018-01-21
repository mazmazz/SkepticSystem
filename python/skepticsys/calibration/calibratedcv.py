from .classifiercv import ClassifierCV
from .euler_sigmoid import _EulerSigmoidCalibration
from .rocch import _ROCCHCalibration
from sklearn.calibration import _CalibratedClassifier, _SigmoidCalibration, IsotonicRegression
from sklearn.base import BaseEstimator
from sklearn.preprocessing import label_binarize, LabelEncoder
from betacal import BetaCalibration, _BetaCal

class CalibratedClassifierCV(ClassifierCV):
    def __init__(self, base_estimator=None, method='sigmoid', cv=3, prefit_callback=None, prefit_params = {}, postfit_callback = None, postfit_params = {}):
        self.method = method
        super().__init__(
            base_estimator=base_estimator
            , cv=cv
            , super_class=_CustomCalibratedClassifier
            , super_params={'method':self.method}
            , prefit_callback=prefit_callback
            , prefit_params=prefit_params
            , postfit_callback=postfit_callback
            , postfit_params=postfit_params
        )
    
class _CustomCalibratedClassifier(_CalibratedClassifier):
    def fit(self, X, y, sample_weight=None):
        """Calibrate the fitted model

        Parameters
        ----------
        X : array-like, shape (n_samples, n_features)
            Training data.

        y : array-like, shape (n_samples,)
            Target values.

        sample_weight : array-like, shape = [n_samples] or None
            Sample weights. If None, then samples are equally weighted.

        Returns
        -------
        self : object
            Returns an instance of self.
        """

        self.label_encoder_ = LabelEncoder()
        if self.classes is None:
            self.label_encoder_.fit(y)
        else:
            self.label_encoder_.fit(self.classes)

        self.classes_ = self.label_encoder_.classes_
        Y = label_binarize(y, self.classes_)

        df, idx_pos_class = self._preproc(X)
        self.calibrators_ = []
        for k, this_df in zip(idx_pos_class, df.T):
            if self.method == 'isotonic':
                calibrator = IsotonicRegression(out_of_bounds='clip')
            elif self.method == 'sigmoid':
                calibrator = _SigmoidCalibration()
            elif self.method == 'euler':
                calibrator = _EulerSigmoidCalibration()
            elif self.method == 'beta':
                calibrator = BetaCalibration()
            elif self.method in ['rocch','convex']:
                calibrator = _ROCCHCalibration()
            elif isinstance(self.method, BaseEstimator):
                calibrator = self.method
            else:
                raise ValueError('method should be "sigmoid" or '
                                 '"isotonic". Got %s.' % self.method)
            calibrator.fit(this_df, Y[:, k], sample_weight)
            self.calibrators_.append(calibrator)

        return self
