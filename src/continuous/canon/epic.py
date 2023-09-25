import numpy as np
from _types import RewardCont, EnvInfoCont
from utils import timed, sample_space
 
def epic_canon_cont(reward: RewardCont,
                    env_info: EnvInfoCont,
                    n_samples: float = 10**6) -> RewardCont:
  """
    C(R)(s,a,s') = R(s,a,s') + E[\gamma R(s',A,S') - R(s,A,S') - \gamma R(S,A,S')]
    S,S' \sim D_s (assume uniform)
    A \sim D_a (assume uniform)
  """

  @timed
  def epic_canonized(s: float, a: float, s_prime: float) -> float:
    samples = []
    for _ in range(n_samples):
      # sample S, A, S' from uniform distributions
      S = sample_space(env_info.state_space)
      A = sample_space(env_info.action_space)
      S_prime = sample_space(env_info.state_space)

      samples.append(reward(s, a, s_prime)
                     + env_info.discount * reward(s_prime, A, S_prime)
                     - reward(s, A, S_prime)
                     - env_info.discount * reward(S, A, S_prime))

    return np.mean(samples)
      
  return epic_canonized
