import numpy as np
from continuous.distance import canon_and_norm_cont
from tests.continuous.toy_env_cont import reward, env_info
from utils import sample_space

def test_canon_and_norm():
    can_norm_dict = canon_and_norm_cont(reward, env_info)

    # check that all the keys are there
    for c in ['VAL', 'EPIC', 'DARD']:
        for n in ['1', '2', 'inf', 'weighted_1', 'weighted_2', 'weighted_inf']:
            assert f'{c}-{n}' in can_norm_dict

    # check that all the values are between 0 and 1
    for func in can_norm_dict.values():
        for _ in range(10):
            s = sample_space(env_info.state_space)
            a = sample_space(env_info.action_space)
            s_prime = sample_space(env_info.state_space)
            assert 0 <= func(s, a, s_prime) <= 1

    raise NotImplementedError
