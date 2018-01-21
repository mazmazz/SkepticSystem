# parent submodules
import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from calibration import ClassifierCV, CalibratedClassifierCV, ThresholdClassifierCV, CutoffClassifierCV
from cross_validation import WindowSplit
sys.path.pop(0)
# end parent submodules

import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.datasets import make_classification
from xgboost import XGBClassifier
import sklearn.metrics as skm

def init():
    X, y = make_classification(7000)

    print('Truth freqs: %s' % str({k: v for k, v in zip(*[x.tolist() for x in np.unique(y, return_counts=True)])}))

    # Base CV
    cv = WindowSplit(test_size=120, step_size=120, sliding_size=1000, initial_test_index=-480)
    base_clf = XGBClassifier()
    cv_clf = ClassifierCV(base_clf, cv)
    cv_clf.fit(X, y)

    print('Base accuracy: %s' % cv_clf.score_cv())
    print('Base logloss: %s' % cv_clf.score_cv(skm.log_loss))

    test_calibration(X, y, cal_method='sigmoid')
    test_calibration(X, y, cal_method='isotonic')
    test_calibration(X, y, cal_method='rocch')
    test_calibration(X, y, cal_method='beta')
    test_threshold(X, y, thr_method='youden')
    test_threshold(X, y, thr_method='roc')
    test_cutoff(X, y)
    test_calibration_threshold(X, y, cal_method='sigmoid', thr_method='youden')
    test_calibration_threshold(X, y, cal_method='isotonic', thr_method='youden')
    test_calibration_threshold(X, y, cal_method='rocch', thr_method='youden')
    test_calibration_threshold(X, y, cal_method='beta', thr_method='youden')
    test_calibration_threshold(X, y, cal_method='sigmoid', thr_method='roc')
    test_calibration_threshold(X, y, cal_method='isotonic', thr_method='roc')
    test_calibration_threshold(X, y, cal_method='rocch', thr_method='roc')
    test_calibration_threshold(X, y, cal_method='beta', thr_method='roc')
    test_calibration_cutoff(X, y, cal_method='sigmoid')
    test_calibration_cutoff(X, y, cal_method='isotonic')
    test_calibration_cutoff(X, y, cal_method='rocch')
    test_calibration_cutoff(X, y, cal_method='beta')

    import pdb; pdb.set_trace()
    pass

def test_calibration(X, y, cal_method='sigmoid'):
    # CalibratedClassifierCV
    base_clf = XGBClassifier(objective='binary:logitraw')
    base_cv = WindowSplit(test_size=120, step_size=120, sliding_size=5480, initial_test_index=-480)
    cal_cv = WindowSplit(test_size=120, step_size=120, sliding_size=5000, initial_test_index=-480)
    cal_clf = CalibratedClassifierCV(base_clf, method=cal_method, cv=cal_cv)
    cv_clf = ClassifierCV(cal_clf, cv=base_cv)
    cv_clf.fit(X, y)

    print(cal_method + ' Calibrated accuracy: %s' % cv_clf.score_cv())
    print(cal_method + ' Calibrated logloss: %s' % cv_clf.score_cv(skm.log_loss))

def test_threshold(X, y, thr_method='youden'):
    # ThresholdClassifierCV
    base_clf = XGBClassifier(objective='binary:logistic')
    base_cv = WindowSplit(test_size=120, step_size=120, sliding_size=5480, initial_test_index=-480)
    thr_cv = WindowSplit(test_size=120, step_size=120, sliding_size=5000, initial_test_index=-480)
    thr_clf = ThresholdClassifierCV(base_clf, method=thr_method, cv=thr_cv)
    cv_clf = ClassifierCV(thr_clf, cv=base_cv)
    cv_clf.fit(X, y)

    print(thr_method + ' Threshold: %s' % (sum([unit.threshold for unit in cv_clf.classifiers_])/len([unit.threshold for unit in cv_clf.classifiers_])))
    print(thr_method + ' Threshold accuracy: %s' % cv_clf.score_cv())
    print(thr_method + ' Threshold logloss: %s' % cv_clf.score_cv(skm.log_loss))

def test_cutoff(X, y):
    # # CutoffClassifierCV
    base_clf = XGBClassifier(objective='binary:logistic')
    base_cv = WindowSplit(test_size=120, step_size=120, sliding_size=5480, initial_test_index=-480)
    cut_cv = WindowSplit(test_size=120, step_size=120, sliding_size=5000, initial_test_index=-480)
    cut_clf = CutoffClassifierCV(base_clf, cv=cut_cv)
    cv_clf = ClassifierCV(cut_clf, cv=base_cv)
    cv_clf.fit(X, y)

    # Get cutoff
    upper_cutoffs = [unit.cutoff[0] for unit in cv_clf.classifiers_]
    lower_cutoffs = [unit.cutoff[1] for unit in cv_clf.classifiers_]
    print('Cutoff: ' + str((sum(upper_cutoffs)/len(upper_cutoffs), sum(lower_cutoffs)/len(lower_cutoffs))))
    print('Cutoff accuracy: %s' % cv_clf.score_cv())
    print('Cutoff logloss: %s' % cv_clf.score_cv(skm.log_loss))

def test_calibration_threshold(X, y, cal_method='sigmoid', thr_method='youden'):
    # # CalibratedClassifierCV and CutoffClassifierCV
    cal_cv = WindowSplit(test_size=120, step_size=120, sliding_size=5000, initial_test_index=-480)
    thr_cv = WindowSplit(test_size=120, step_size=120, sliding_size=5480, initial_test_index=-480)
    cv_cv = WindowSplit(test_size=120, step_size=120, sliding_size=5960, initial_test_index=-480)
    base_clf = XGBClassifier(objective='binary:logitraw')
    cal_clf = CalibratedClassifierCV(base_clf, method=cal_method, cv=cal_cv)
    thr_clf = ThresholdClassifierCV(cal_clf, method=thr_method, cv=thr_cv)
    cv_clf = ClassifierCV(thr_clf, cv=cv_cv)
    cv_clf.fit(X, y)

    print(cal_method + ' ' + thr_method + ' Threshold: %s' % (sum([unit.threshold for unit in cv_clf.classifiers_])/len([unit.threshold for unit in cv_clf.classifiers_])))
    print(cal_method + ' ' + thr_method + ' Threshold accuracy: %s' % cv_clf.score_cv())
    print(cal_method + ' ' + thr_method + ' Threshold logloss: %s' % cv_clf.score_cv(skm.log_loss))

def test_calibration_cutoff(X, y, cal_method='sigmoid'):
    # # CalibratedClassifierCV and CutoffClassifierCV
    cal_cv = WindowSplit(test_size=120, step_size=120, sliding_size=5000, initial_test_index=-480)
    cut_cv = WindowSplit(test_size=120, step_size=120, sliding_size=5480, initial_test_index=-480)
    cv_cv = WindowSplit(test_size=120, step_size=120, sliding_size=5960, initial_test_index=-480)
    base_clf = XGBClassifier(objective='binary:logitraw')
    cal_clf = CalibratedClassifierCV(base_clf, method=cal_method, cv=cal_cv)
    cut_clf = CutoffClassifierCV(cal_clf, cv=cut_cv)
    cv_clf = ClassifierCV(cut_clf, cv=cv_cv)
    cv_clf.fit(X, y)

    # Get cutoff
    upper_cutoffs = [unit.cutoff[0] for unit in cv_clf.classifiers_]
    lower_cutoffs = [unit.cutoff[1] for unit in cv_clf.classifiers_]
    print(cal_method + ' Cutoff: ' + str((sum(upper_cutoffs)/len(upper_cutoffs), sum(lower_cutoffs)/len(lower_cutoffs))))
    print(cal_method + ' Cutoff accuracy: %s' % cv_clf.score_cv())
    print(cal_method + ' Cutoff logloss: %s' % cv_clf.score_cv(skm.log_loss))

if __name__ == '__main__':
    init()
