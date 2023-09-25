import numpy as np
from _types import RewardCont, EnvInfoCont
from utils import sample_space

def consistency_test(canonicalized: RewardCont,
                     env_info: EnvInfoCont,
                     n_transitions: int = 10,
                     n_canon_runs: int = 3,
                     r_thresh: float = 1e-2):
  """
    Test that the canonicalized reward is consistent when run multiple times.
  """

  for _ in range(n_transitions): # pick 10 random transitions
    s = sample_space(env_info.state_space)
    a = sample_space(env_info.action_space)
    s_prime = sample_space(env_info.state_space)

    canon_trans_reward = canonicalized(s, a, s_prime)
    for _ in range(n_canon_runs): # run 3 more times and make sure the results are close
      val2 = canonicalized(s, a, s_prime)
      assert abs(canon_trans_reward - val2) < r_thresh * abs(canon_trans_reward)
