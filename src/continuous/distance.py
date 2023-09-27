from typing import Union, Dict
from continuous.canon.val import val_canon_cont
from continuous.canon.epic import epic_canon_cont
from continuous.canon.dard import dard_canon_cont
from continuous.norm import norm_cont
from _types import RewardCont, EnvInfoCont
from utils import timed
 
canon_funcs = {
    'VAL': val_canon_cont,
    'EPIC': epic_canon_cont,
    'DARD': dard_canon_cont,
}

@timed
def canon_and_norm_cont(reward: RewardCont,
                        env_info: EnvInfoCont,
                        norm_opts: Union[int, float] = [1, 2, float('inf')],
                        n_canon_samples: int = 10**6,
                        n_norm_samples: int = 10**3) -> Dict[str, RewardCont]:
    """
    Returns a dictionary of all the possible canonicalizations and normalizations
    (lists of possible options are defined in as constants in this file).
    """
    can_r = {c_name: canon_funcs[c_name](reward, env_info, n_canon_samples)
            for c_name in canon_funcs.keys()}
    
    norm_r = {}
    for c_name, val in can_r.items():
        for n_ord in norm_opts:
            def normalized(s: float, a: float, s_prime: float) -> float:
                return val(s, a, s_prime) / norm_cont(val,
                                                      env_info.state_space,
                                                      env_info.action_space,
                                                      n_ord,
                                                      n_norm_samples)
                                                      
            norm_r[f'{c_name}-{n_ord}'] = normalized
    
    return norm_r
