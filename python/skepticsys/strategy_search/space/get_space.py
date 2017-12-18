from .data import get_data_space
from .indicator import get_indicator_space
from .classifier import get_classifier_space
from .calibration import get_calibration_space

def get_space(args):
    spaces = [
        get_data_space(**(args['data__args'] if 'data__args' in args else {}))
        , get_indicator_space(**(args['indicator__args'] if 'indicator__args' in args else {}))
        , get_classifier_space()
        , get_calibration_space()
    ]

    return {k: v for d in spaces for k, v in d.items()}
