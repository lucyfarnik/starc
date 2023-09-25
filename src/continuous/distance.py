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
norm_opts = [1, 2, float('inf'), 'weighted_1', 'weighted_2', 'weighted_inf']

@timed
def canon_and_norm_cont(reward: RewardCont,
                        env_info: EnvInfoCont) -> dict[str, RewardCont]:
    """
    Returns a dictionary of all the possible canonicalizations and normalizations
    (lists of possible options are defined in as constants in this file).
    """
    can_r = {c_name: canon_funcs[c_name](reward, env_info)
            for c_name in canon_funcs.keys()}
    
    norm_r = {}
    for c_name, val in can_r.items():
        for n_ord in norm_opts:
            def normalized(s: float, a: float, s_prime: float) -> float:
                return val(s, a, s_prime) / norm_cont(val, env_info, n_ord)
                                                      
            norm_r[f'{c_name}-{n_ord}'] = normalized
    
    return norm_r
