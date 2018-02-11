# Data Retrieval Spec

## Sample len

Either a single number or a dict specifying train, test, and target
lengths. Target length is the absolute value sum of start_offset 
(usually -1) and end_offset. `sample_len` is the sum of all three values.

## Test Bounds

`from_test`: `start_index` and `end_index` are based on the test split,
while total data bounds (train and target bounds) are interpreted 
relative to test bounds.

NOTE: In `candidate.py`, test split is rolled into nominal train split, 
while verification split is passed as the nominal test split.

To calculate real test split, total data bounds must accurately
encompass `train_len + test_n * test_len + target_len` where `test_n` 
is the total allowed test splits, no more no less. 

Each split counts backwards from `len(data) - target_len - test_n * test_len`
, where each test split is of length `test_len`. Any remaining data 
after `test_n` is training data.

### Ways to pass index bounds

`start_index`: Starting index value of data, inclusive. 
    * If passed alone, data bounds are always based on the start index.

`end_index`: Ending index value of data, non-inclusive.
    * If passed alone, data bounds are based on the end index; train and
      test bounds are counted backwards from the end index.
    * If passed with `start_index`, this is a limiter. If test length
      is lower than `end_index - start_index`, data is truncated to
      test length. Else, exception is thrown for not enough data.
      