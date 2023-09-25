import numpy as np
from _types import RewardCont, EnvInfoCont
from typing import Union
from utils import timed, sample_space
 
@timed
def norm_cont(reward: RewardCont,
              env_info: EnvInfoCont,
              ord: Union[int, float, str] = 2,
              n_samples: int = 10**3) -> float:
  """
    Takes the norm of a continuous function
    ord: 0, 1, 2, 'weighted_1', 'weighted_2', 'weighted_inf', 'inf'
  """

  if ord == 0: return 1 # baseline (no norm)

  # is_weighted = type(ord) is str and 'weighted' in ord
  # norm_ord = float(ord.split('_')[1]) if is_weighted else float(ord)

  # L_infty norm — find the maximum
  if ord == float('inf'):
    # do random sampling to find the maximum
    # TODO try assuming the function is convex and differentiable in Torch; do optimization
    max_sample = float('-inf')
    for _ in range(n_samples):
      # sample s, a, s'
      s = sample_space(env_info.state_space)
      a = sample_space(env_info.action_space)
      s_prime = sample_space(env_info.state_space)

      sample_val = abs(reward(s, a, s_prime)) # |r(s,a,s')|
      # if is_weighted: sample_val *= env_info.trans_prob(s, a, s_prime) # weighted norms

      if sample_val > max_sample:
        max_sample = sample_val
    
    return max_sample

  sample_sum = 0 
  for _ in range(n_samples):
    # sample s, a, s'
    s = sample_space(env_info.state_space)
    a = sample_space(env_info.action_space)
    s_prime = sample_space(env_info.state_space)

    sample_val = abs(reward(s, a, s_prime))**ord # |r(s,a,s')|^p
    # if is_weighted: sample_val *= env_info.trans_prob(s, a, s_prime) # weighted norms

    sample_sum += sample_val
  
  # volume of the domain, ie the integral of 1 over the domain
  state_space_volume = 1
  for interval in env_info.state_space:
    state_space_volume *= interval[1] - interval[0]
  action_space_volume = 1
  for interval in env_info.action_space:
    action_space_volume *= interval[1] - interval[0]
  domain_volume = (state_space_volume ** 2) * action_space_volume

  # (V / N * sum)^(1/p)
  # TODO check that including the volume in this way is the "normal" Lp norm
  # TODO (though it's definitely bilipschitz equivalent to it so doesn't really matter)
  return (domain_volume / n_samples * sample_sum)**(1/ord)
  