# parent submodules
import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from datasets import load_prices, get_target
from cross_validation import WindowSplit
from optimizer.candidate import do_data, get_sample_len, get_cv, get_transforms
from preprocessors import NanSampler
sys.path.pop(0)
# end parent submodules

import numpy as np
import pandas as pd
import unittest

# Tests:
# Start/End/Start-End/No Index
# Verify/No Verify
# Post/No Post
## Post-data exists?

class TestDataMethods(unittest.TestCase):
    def setUp(self):
        self.data_params = {
            'instrument': 'USDJPY'
            , 'granularity': 'H1'
            , 'source': 'csv'
            , 'dir': 'D:\\Projects\\Prices'
            , 'start_target': -1
            , 'end_target': -60
            , 'start_index': None
            , 'end_index': None
            , 'start_buffer': 1000
            , 'end_buffer': 0
        }

        self.cv_params = {
            'train_size': 2000
            , 'train_sliding': False # False

            # master
            , 'test_size': 100
            , 'test_n': 4

            , 'verify_factor': [0.75,0.5,0.25]
            
            , 'transforms': [
                {
                    'calibration': True
                    , 'method': 'sigmoid'
                    , 'test_size': 75
                    , 'test_n': 2
                }
                , {
                    'cutoff': True
                    , 'test_size': 22
                    , 'test_n': 3
                }
            ]
        }

    def test_startindex(self):
        data_params = {**self.data_params, 'start_index': 201701030000}
        cv_params = {**self.cv_params}

    def test_endindex(self):
        data_params = {**self.data_params, 'end_index': 201701030000}
        cv_params = {**self.cv_params}
        self.metatest_samplelen(data_params, cv_params)
        self.metatest_dataretrieval(data_params, cv_params)
        self.metatest_cvsplit(data_params, cv_params)

    def test_startendindex(self):
        data_params = {**self.data_params, 'start_index': 201701030000, 'end_index': 201701060000}
        cv_params = {**self.cv_params}

    def test_noindex(self):
        data_params = {**self.data_params}
        cv_params = {**self.cv_params}

    def metatest_samplelen(self, data_params, cv_params):
        transforms = get_transforms(cv_params)

        sample_len = get_sample_len(data_params, cv_params)

        self.assertEqual(sample_len['train']
            , sum([cv_params['train_size'], *[t['test_size'] * t['test_n'] for t in transforms]]) # 2616
            , msg='sample_len train size incorrect')
        
        self.assertEqual(sample_len['test']
            , cv_params['test_size'] * cv_params['test_n'] * max(cv_params['verify_factor']) # 300
            , msg='sample_len test size incorrect')

        self.assertEqual(sample_len['post']
            , cv_params['test_size'] * (cv_params['test_n'] - cv_params['test_n'] * min(cv_params['verify_factor'])) # 300
            , msg='sample_len post size incorrect')

        self.assertEqual(sample_len['target']
            , abs(data_params['end_target']) # 60
            , msg='sample_len target size incorrect')

    def metatest_dataretrieval(self, data_params, cv_params):
        transforms = get_transforms(cv_params)
        sample_len = get_sample_len(data_params, cv_params)

        prices_params = {**data_params}
        prices_params.pop('start_target')
        prices_params.pop('end_target')
        prices = load_prices(**prices_params
            , sample_len=sample_len
            , from_test=True
        )
        target = get_target(prices, data_params['end_target'], start_offset=data_params['start_target'])

        self.assertEqual(len(prices)
            , sum(sample_len.values()) + data_params['start_buffer'] + data_params['end_buffer'] # 4276
            , msg='prices len incorrect to sample_len')

        if data_params['end_index'] is not None:
            if data_params['start_index'] is not None:
                # Start and end indexes
                raise NotImplementedError('End index and Start index') 

            else:
                # End index only
                self.assertEqual(prices.index.get_loc(data_params['end_index'])
                    , len(prices) - data_params['end_buffer'] - sample_len['target'] - sample_len['post'] # 3916
                    , msg='prices end_index not in correct loc')

                self.assertEqual(data_params['start_buffer'] + sample_len['train'] + sample_len['test']
                    , prices.index.get_loc(data_params['end_index']) # 3916
                    , msg='prices start_buffer, train, and test not equal to end_index loc')

        elif data_params['start_index'] is not None:
            # Start index only
            raise NotImplementedError('Start index only')

        else:
            # No indexes
            raise NotImplementedError('No indexes')
        
        nans = NanSampler(drop_inf=False)
        prices_model, target_model = nans.sample(prices, target)

        self.assertEqual(len(prices_model)
            , len(prices) - abs(data_params['end_target']) # 4216
            , msg='prices_model post-NaN incorrect length')

    def metatest_cvsplit(self, data_params, cv_params):
        transforms = get_transforms(cv_params)
        sample_len = get_sample_len(data_params, cv_params)

        prices_params = {**data_params}
        prices_params.pop('start_target')
        prices_params.pop('end_target')
        prices = load_prices(**prices_params
            , sample_len=sample_len
            , from_test=True
        )
        target = get_target(prices, data_params['end_target'], start_offset=data_params['start_target'])

        nans = NanSampler(drop_inf=False)
        prices_model, target_model = nans.sample(prices, target)

        cv = get_cv(prices_model, data_params, cv_params)
        cv_base = get_cv(prices_model, data_params, cv_params, base_only=True)
        cv_verify = get_cv(prices_model, data_params, cv_params, do_verify=True)

        last_cv = WindowSplit(delay_size=0, final_index=None, force_sliding_min=True,
            initial_test_index=-300, initial_train_index=0, min_test_size=None,
            sliding_size=2216, step_size=100, test_remainder=False,
            test_size=100)
        last_cv_list = list(last_cv.split(prices_model))

        import pdb; pdb.set_trace()
        pass

if __name__ == '__main__':
    unittest.main()
