from hyperopt import hp

def get_calibration_space(
    calibration = [None,'euler','sigmoid','beta','isotonic']
    , threshold = [None,'topleft','youden']
    , cutoff = [None,'accuracy']
):
    spaces = []

    if bool(calibration):
        spaces.append(get_calibration_params(calibration))

    if bool(threshold):
        spaces.append(get_threshold_params(threshold))

    if bool(cutoff):
        spaces.append(get_cutoff_params(cutoff))

    return {k: v for d in spaces for k, v in d.items()}

def get_calibration_params(calibration):
    calibration_types = []

    if 'euler' in calibration:
        calibration_types.append({'type': 'euler'})
    
    if 'sigmoid' in calibration:
        calibration_types.append({'type': 'sigmoid'})

    if 'beta' in calibration:
        calibration_types.append({'type': 'beta'})

    if 'isotonic' in calibration:
        calibration_types.append({'type': 'isotonic'})

    choices = []

    if len(calibration_types) > 0:
        if None in calibration:
            choices.append(None)
        choices.append({
            'cv_lookback': hp.choice('calibration__cv_lookback', range(10, 101, 10))
            , 'type': hp.choice('calibration__type', calibration_types)
        })

    return {
        'calibration__params': hp.choice('calibration__params', choices) if len(choices) > 0 else None
    }

def get_threshold_params(threshold):
    threshold_types = []

    if 'topleft' in threshold:
        threshold_types.append({'type': 'topleft'})
    
    if 'youden' in threshold:
        threshold_types.append({'type': 'youden'})

    choices = []

    if len(threshold_types) > 0:
        if None in threshold:
            choices.append(None)
        choices.append({
            'cv_lookback': hp.choice('threshold__cv_lookback', range(10, 101, 10))
            , 'type': hp.choice('threshold__type', threshold_types)
        })

    return {
        'threshold__params': hp.choice('threshold__params', choices) if len(choices) > 0 else None
    }

def get_cutoff_params(cutoff):
    cutoff_types = []

    if 'accuracy' in cutoff:
        cutoff_types.append({'type': 'accuracy'})

    choices = []

    if len(cutoff_types) > 0:
        if None in cutoff:
            choices.append(None)
        choices.append({
            'cv_lookback': hp.choice('cutoff__cv_lookback', range(10, 101, 10))
            , 'type': hp.choice('cutoff__type', cutoff_types)
        })

    return {
        'cutoff__params': hp.choice('cutoff__params', choices) if len(choices) > 0 else None
    }
