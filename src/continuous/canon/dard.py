import numpy as np
from _types import RewardCont, EnvInfoCont
from utils import timed, sample_space
 
def dard_canon_cont(reward: RewardCont,
                    env_info: EnvInfoCont,
                    n_samples: float = 10**6) -> RewardCont:
  """
    C(R)(s,a,s') = R(s,a,s') + E[\gamma R(s',A,S'') - R(s,A,S') - \gamma R(S',A,S'')]
    A \sim D_a (assume uniform)
    S' \sim T(S'|s,A)
    S'' \sim T(S''|s',A)
  """

  @timed
  def dard_canonized(s: float, a: float, s_prime: float) -> float:
    samples = []
    for _ in range(n_samples):
      # sample S, A, S' from uniform distributions
      A = sample_space(env_info.action_space)
      S_prime = env_info.trans_dist(s, A)
      S_double = env_info.trans_dist(s_prime, A)

      samples.append(reward(s, a, s_prime)
                     + env_info.discount * reward(s_prime, A, S_double)
                     - reward(s, A, S_prime)
                     - env_info.discount * reward(S_prime, A, S_double))

    return np.mean(samples)
      
  return dard_canonized
