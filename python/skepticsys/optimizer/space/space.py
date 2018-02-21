from .data import get_data_space
from .indicator import get_indicator_space
from .classifier import get_classifier_space
from .cv import get_cv_space

def get_space(args={}, do_transforms=False):
    data_args, cv_args = process_test_args(
        args['data__args'] if 'data__args' in args else {}
        , args['cv__args'] if 'cv__args' in args else {}
        , **(args['test__args'] if 'test__args' in args else {})
    )

    spaces = [
        get_data_space(**data_args)
        , get_indicator_space(args['indicator__args'] if 'indicator__args' in args else {})
        , get_classifier_space(args['classifier__args'] if 'classifier__args' in args else {})
        , get_cv_space(cv_args, do_transforms=do_transforms)
        , {'meta__params': {**args['meta__args']}}
    ]

    return {k: v for d in spaces for k, v in d.items()}

def process_test_args(
    data_args, cv_args
    , start_index = None
    , end_index = None
    , end_target = None # -61
    , test_size = None # 120
    , test_n = None # 4
    , train_size = None # 6000
    , train_sliding = None # True
):
    # we do this so that test args can have their own YAML config
    data_args = {**data_args}
    if start_index is not None: data_args['start_index'] = start_index
    if end_index is not None: data_args['end_index'] = end_index
    if end_target is not None: data_args['end_target'] = end_target

    cv_args = {**cv_args}
    if test_size is not None: cv_args['test_size'] = test_size
    if test_n is not None: cv_args['test_n'] = test_n
    if train_size is not None: cv_args['train_size'] = train_size
    if train_sliding is not None: cv_args['train_sliding'] = train_sliding

    return data_args, cv_args
