import numpy as np
from functools import partial
from _types import RewardCont, EnvInfoCont
from utils import timed

# @timed
def _val_canonicalized(reward: RewardCont,
                       env_info: EnvInfoCont,
                       n_samples: int,
                       s: float,
                       a: float,
                       s_prime: float) -> float:
  if env_info.trans_dist_deterministic and env_info.state_vals_deterministic:
    S_prime = env_info.trans_dist(s, a)
    return reward(s, a, S_prime) - \
            env_info.state_vals(s) + \
            env_info.discount * env_info.state_vals(S_prime)
  
  samples = []
  for _ in range(n_samples):
    # sample S_prime from trans_dist
    S_prime = env_info.trans_dist(s, a)

    samples.append(reward(s, a, S_prime)
                    - env_info.state_vals(s)
                    + env_info.discount * env_info.state_vals(S_prime))

  return np.mean(samples)

def val_canon_cont(reward: RewardCont,
                   env_info: EnvInfoCont,
                   n_samples: int = 10**6) -> RewardCont:
  """
    C(R)(s,a,s') = E[R(s,a,S') + gamma*V^\pi(S') - V^\pi(s)]
    S' ~ T(S' | s,a)
  """

  return partial(_val_canonicalized, reward, env_info, n_samples)
