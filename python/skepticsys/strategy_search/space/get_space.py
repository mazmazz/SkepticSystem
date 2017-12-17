from .data import get_data_space
from .indicator import get_indicator_space
from .classifier import get_classifier_space
from .calibration import get_calibration_space

def get_space(args):
    spaces = [
        get_data_space(**args['data__args'])
        , get_indicator_space()
        , get_classifier_space()
        , get_calibration_space()
    ]

    return {k: v for d in spaces for k, v in d.items()}
