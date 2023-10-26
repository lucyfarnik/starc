import numpy as np
from functools import partial
from _types import RewardCont, EnvInfoCont
from utils import timed, sample_space

# @timed
def _dard_canonicalized(reward: RewardCont,
                        env_info: EnvInfoCont,
                        n_samples: int,
                        s: float,
                        a: float,
                        s_prime: float) -> float:
  """
    Exectues the DARD canonicalized version of the reward function on a specific
    transition and returns the reward value
  """
  if env_info.trans_dist_deterministic:
    n_samples_adjusted = n_samples // 10 # instead of 22 dimensions, we only sample 2
  else:
    n_samples_adjusted = n_samples
  
  samples_sum = 0
  for _ in range(n_samples_adjusted):
    # sample S, A, S' from uniform distributions
    A = sample_space(env_info.action_space)
    S_prime = env_info.trans_dist(s, A)
    S_double = env_info.trans_dist(s_prime, A)

    samples_sum += reward(s, a, s_prime) + \
                    env_info.discount * reward(s_prime, A, S_double) - \
                    reward(s, A, S_prime) - \
                    env_info.discount * reward(S_prime, A, S_double)

  return samples_sum / n_samples_adjusted

def dard_canon_cont(reward: RewardCont,
                    env_info: EnvInfoCont,
                    n_samples: int = 10**6) -> RewardCont:
  """
    Returns the DARD canonicalized reward function

    C(R)(s,a,s') = R(s,a,s') + E[\gamma R(s',A,S'') - R(s,A,S') - \gamma R(S',A,S'')]
    A \sim D_a (assume uniform)
    S' \sim T(S'|s,A)
    S'' \sim T(S''|s',A)
  """
   
  return partial(_dard_canonicalized, reward, env_info, n_samples)
