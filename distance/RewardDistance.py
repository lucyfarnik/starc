import numpy as np
from distance.canon import canon_funcs
from typing import Literal
from env import Env
from _types import Reward

# Class for calculating EPIC-like distances
# When instantiating, pass canonicalization choice (as string), and which
# norms should be used for both normalization and distances.
# eg. dist = RewardDistance('EPIC', float('inf'), 1)
# Then call it with two rewards and the environment they're for and it'll
# give you the distance
class RewardDistance():
  def __init__(
    self,
    canon_func: Literal['EPIC', 'DARD'],
    norm_ord: int|float = 2,
    dist_ord: int|float = 2,
  ):
    self.canon_func = canon_func
    self.norm_ord = norm_ord
    self.dist_ord = dist_ord

  def __call__(self, r1: Reward, r2: Reward, env: Env) -> float:
    # canonicalize
    can1 = canon_funcs[self.canon_func](r1, env)
    can2 = canon_funcs[self.canon_func](r2, env)

    # normalize
    standard1 = can1 / np.linalg.norm(can1.flatten(), self.norm_ord)
    standard2 = can2 / np.linalg.norm(can2.flatten(), self.norm_ord)

    # distance
    return np.linalg.norm((standard1 - standard2).flatten(), self.dist_ord)
