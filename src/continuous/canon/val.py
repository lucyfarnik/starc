import numpy as np
from _types import RewardCont, EnvInfoCont
from utils import timed
 
def val_canon_cont(reward: RewardCont,
                   env_info: EnvInfoCont,
                   n_samples: float = 10**6) -> RewardCont:
  """
    C(R)(s,a,s') = E[R(s,a,S') + gamma*V^\pi(S') - V^\pi(s)]
    S' ~ T(S' | s,a)
  """

  @timed
  def val_canonized(s: float, a: float, s_prime: float) -> float:
    samples = []
    for _ in range(n_samples):
      # sample S_prime from trans_dist
      S_prime = env_info.trans_dist(s, a)

      samples.append(reward(s, a, S_prime)
                     - env_info.state_vals(s)
                     + env_info.discount * env_info.state_vals(S_prime))

    return np.mean(samples)
      
  return val_canonized
