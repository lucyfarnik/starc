import numpy as np
from functools import partial
from _types import RewardCont, EnvInfoCont
from utils import timed, sample_space

# @timed
def _epic_canonicalized(reward: RewardCont,
                        env_info: EnvInfoCont,
                        n_samples: int,
                        s: float,
                        a: float,
                        s_prime: float) -> float:
  """
    Executes the EPIC canonicalized version of the reward function and returns
    its value for the transition
  """
  samples_sum = 0
  for _ in range(n_samples):
    # sample S, A, S' from uniform distributions
    S = sample_space(env_info.state_space)
    A = sample_space(env_info.action_space)
    S_prime = sample_space(env_info.state_space)

    samples_sum += reward(s, a, s_prime) + \
                    env_info.discount * reward(s_prime, A, S_prime) - \
                    reward(s, A, S_prime) - \
                    env_info.discount * reward(S, A, S_prime)

  return samples_sum / n_samples
      
def epic_canon_cont(reward: RewardCont,
                    env_info: EnvInfoCont,
                    n_samples: int = 10**6) -> RewardCont:
  """
    Returns the EPIC canonicalized reward function

    C(R)(s,a,s') = R(s,a,s') + E[\gamma R(s',A,S') - R(s,A,S') - \gamma R(S,A,S')]
    S,S' \sim D_s (assume uniform)
    A \sim D_a (assume uniform)
  """

  return partial(_epic_canonicalized, reward, env_info, n_samples)
