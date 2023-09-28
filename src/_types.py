import numpy as np
from typing import Callable, List, Tuple
from dataclasses import dataclass

# just here for readability in the other files
Reward = np.ndarray # [S, A, S']; reward at transition s -a-> s'
Policy = np.ndarray # [S] action to take in state; deterministic

# continuous stuff
RewardCont = Callable[[float, float, float], float] # S, A, S' -> R
TransCont = Callable[[float, float], float] # S, A -> S'
StateValCont = Callable[[float], float] # S -> V
Space = List[Tuple[float, float]] # list of (min, max); inclusive on both sides

@dataclass
class EnvInfoCont():
  trans_dist: TransCont
  trans_dist_deterministic: bool
  discount: float
  state_space: Space
  action_space: Space
  state_vals: StateValCont
  state_vals_deterministic: bool
