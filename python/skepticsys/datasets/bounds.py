def get_bounds(
    start_loc
    , end_loc
    , sample_len
    , max_bounds
    , min_bounds=0
    , from_test=False
    , start_buffer=1000
    , end_buffer=0
    , start_index=None
    , end_index=None
    , target_gap=False
):
    train_len, test_len, post_len, target_len, sample_len = split_sample_len(sample_len)

    if from_test:
        # get start bounds
        if sample_len is not None:
            if start_index is not None:
                if train_len is None:
                    raise ValueError('Train length must be specified separately if start_index is specified.')

                query_loc = start_loc - train_len - start_buffer

                if target_gap:
                    query_loc -= target_len

                if query_loc < min_bounds:
                    raise ValueError('Not enough data for requested train_len %s and start_index' % (train_len, start_index))
            else:
                query_loc = end_loc - sample_len - start_buffer + post_len # offset post_len from sample_len

                if target_gap:
                    query_loc -= target_len

                if end_index is not None:
                    query_loc += target_len # end_loc already accounts for this

                if query_loc < start_loc:
                    raise ValueError('Not enough data for requested sample_len - post_len %s' % sample_len-post_len)

                start_loc = query_loc + train_len + start_buffer # carries more meaning than original, reflects test bound ending

            final_start_loc = query_loc
        else:
            final_start_loc = start_loc

        # get end bounds, non-inclusive
        # end bounds must truncate to last available full test split
        if target_len is not None:
            if end_index is not None:
                if start_index is not None:
                    query_loc = start_loc + test_len
                    if query_loc > end_loc:
                        raise ValueError('Not enough data for requested test_len %s' % (test_len))
                    query_loc += target_len
                    if query_loc > max_bounds:
                        raise ValueError('Not enough data for requested test_len %s and target_len %s' % (test_len, target_len))
                    query_loc += post_len
                    if query_loc > max_bounds:
                        raise ValueError('Not enough data for requested test_len %s and target_len %s and post_len %s' % (test_len, target_len, post_len))
                    query_loc += end_buffer
                    if query_loc > max_bounds:
                        raise ValueError('Not enough data for requested test_len %s and target_len %s and post_len %s and end_buffer %s' % (test_len, target_len, post_len, end_buffer))
                    end_loc = query_loc-target_len-post_len-end_buffer # carries more meaning than original, reflects test bound ending
                else:
                    query_loc = end_loc + target_len + post_len + end_buffer
                    if query_loc > max_bounds:
                        raise ValueError('Not enough data for requested test_len %s and target_len %s and post_len %s and end_buffer %s' % (test_len, target_len, post_len, end_buffer))
            else:
                if start_index is not None:
                    query_loc = start_loc + test_len + target_len + post_len + end_buffer
                    if query_loc > max_bounds:
                        raise ValueError('Not enough data for requested test_len %s and target_len %s and post_len %s and end_buffer %s' % (test_len, target_len, post_len, end_buffer))
                else:
                    query_loc = end_loc + post_len + end_buffer # end_loc already accounts for target_len ### todo: test?
                end_loc = query_loc-target_len-post_len-end_buffer # carries more meaning than original, reflects test bound ending
            final_end_loc = query_loc
        else:
            final_end_loc = end_loc

        assert (end_loc-start_loc == test_len+(target_len if target_gap else 0)
                and final_end_loc-end_loc == target_len+post_len+end_buffer
                and start_loc-final_start_loc == train_len+start_buffer
                )
    else:
        # dumb bounds
        if sample_len is not None and sample_len != 0:
            if sample_len > 0:
                final_end_loc = start_loc+sample_len
                final_start_loc = start_loc
            else: 
                final_start_loc = end_loc-sample_len
                final_end_loc = end_loc
        else:
            final_start_loc, final_end_loc = start_loc, end_loc

    return final_start_loc, final_end_loc, start_loc, end_loc

def split_sample_len(sample_len, target_gap=False):
    if isinstance(sample_len, dict):
        train_len = int(abs(sample_len['train']))
        test_len = int(abs(sample_len['test']))
        post_len = int(abs(sample_len['post']))
        target_len = int(abs(sample_len['target']))
        sample_len = int(sum(sample_len.values())) + (target_len if target_gap else 0)
    else:
        train_len, test_len, post_len, target_len = None, None, None, None
    
    return train_len, test_len, post_len, target_len, sample_len
