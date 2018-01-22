from hyperopt import hp
from hyperopt.pyll.base import scope

# parent imports
import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))) # 2 levels up
from cross_validation import WindowSplit
sys.path.pop(0)
# end parent imports

def get_cv_space(params={}):
    space = {**default_cv, **params}
    return {
        'cv__params': space
    }

default_cv = {
    'cv': WindowSplit
    , 'params': {
        'test_size': 120
        , 'step_size': 120
        , 'sliding_size': scope.int(hp.quniform('cv__windowsplit__sliding_size', 480, 6000, 1)) # 960
        , 'initial_test_index': -480 # initial index of TEST series #-480
    }
    #, 'single_split': 100
}
