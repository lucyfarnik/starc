import numpy as np
from typing import Callable
from dataclasses import dataclass

# just here for readability in the other files
Reward = np.ndarray # [S, A, S']; reward at transition s -a-> s'
Policy = np.ndarray # [S] action to take in state; deterministic

# continuous stuff
RewardCont = Callable[[float, float, float], float] # S, A, S' -> R
TransCont = Callable[[float, float], float] # S, A -> S'
TransProbCont = Callable[[float, float, float], float] # S, A, S' -> P(S' | S, A)
StateValCont = Callable[[float], float] # S -> V
Space = list[tuple[float, float]] # list of (min, max); inclusive on both sides

@dataclass
class EnvInfoCont():
  trans_dist: TransCont
  trans_prob: TransProbCont
  discount: float
  state_space: Space
  action_space: Space
  state_vals: StateValCont
