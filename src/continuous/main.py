# Continuous experiment
#! Right now this is just a placeholder
from multiprocessing import Pool
import numpy as np
from continuous.distance import canon_and_norm_cont
from continuous.norm import norm_cont
from _types import EnvInfoCont, RewardCont

if __name__ == '__main__':
  np.random.seed(0)

  env_info: EnvInfoCont = None

  for reward_i in range(64):
    r1: RewardCont = None
    r2: RewardCont = None

    pi1_optimal = None
    pi1_pessimal = None

    canonicalized1 = canon_and_norm_cont(r1, env_info)

    for r_i in interpolate_cont(r1, r2, 16):
      