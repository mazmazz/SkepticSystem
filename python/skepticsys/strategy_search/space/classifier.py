from hyperopt import hp

def get_classifier_space(
    classifiers=['xgb']
):
    if not bool(classifiers):
        return {}

    choices = []

    if 'xgb' in classifiers:
        choices.append({
            # xgb: https://www.analyticsvidhya.com/blog/2016/03/complete-guide-parameter-tuning-xgboost-with-codes-python/
            # https://cambridgespark.com/content/tutorials/hyperparameter-tuning-in-xgboost/index.html
            # https://github.com/dmlc/xgboost/blob/master/doc/parameter.md
            'type': 'xgb'
            , 'max_depth': hp.choice('xgb__max_depth', range(3,11))
            , 'learning_rate': 0.3
            , 'objective': 'binary:logistic' # when testing calibration, will need to detect when to replace this
            , 'eval_metric': hp.choice('xgb__eval_metric', ['error','auc','rmse','logloss'])
                # error allows for threshold different than 0.5 according to error@t. How to roll this in?
            , 'silent': True
        })

    if None in classifiers and len(choices) > 0:
        choices.append(None)

    return {
        'classifier__params': hp.choice('classifier__params', choices) if len(choices) > 0 else None
    }

