from .data import get_data_space

from .classifier import get_classifier_space
from .calibration import get_calibration_space

def get_space():
    spaces = [
        get_data_space()
        , get_classifier_space()
        , get_calibration_space()
    ]

    return {k: v for d in spaces for k, v in d.items()}
