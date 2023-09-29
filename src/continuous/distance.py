from typing import List, Union, Dict
from functools import partial
from continuous.canon.val import val_canon_cont
from continuous.canon.epic import epic_canon_cont
from continuous.canon.dard import dard_canon_cont
from continuous.norm import norm_cont
from _types import RewardCont, EnvInfoCont, Space
from utils import timed
 
canon_funcs = {
    'VAL': val_canon_cont,
    'EPIC': epic_canon_cont,
    'DARD': dard_canon_cont,
}

def _normalized_reward(canonicalized: RewardCont,
                       norm_val: float,
                       s: float,
                       a: float,
                       s_prime: float) -> float:
    if norm_val == 0:
        return canonicalized(s, a, s_prime)
    return canonicalized(s, a, s_prime) / norm_val

# @timed
def canon_and_norm_cont(reward: RewardCont,
                        env_info: EnvInfoCont,
                        canon_func_keys: List[str] = ['VAL', 'EPIC', 'DARD'],
                        norm_opts: Union[int, float] = [1, 2, float('inf')],
                        n_canon_samples: int = 10**6,
                        n_norm_samples: int = 10**3) -> Dict[str, RewardCont]:
    """
    Returns a dictionary of all the possible canonicalizations and normalizations
    (lists of possible options are defined in as constants in this file).
    """
    can_r = {c_name: canon_funcs[c_name](reward, env_info, n_canon_samples)
            for c_name in canon_func_keys}
   
    norm_r = {}
    for c_name, val in can_r.items():
        for n_ord in norm_opts:
            norm_val = norm_cont(val,
                                 env_info.trans_dist,
                                 env_info.state_space,
                                 env_info.action_space,
                                 n_ord,
                                 n_norm_samples)
            normalized = partial(_normalized_reward, val, norm_val)
                                                      
            norm_r[f'{c_name}-{n_ord}'] = normalized
    
    return norm_r
