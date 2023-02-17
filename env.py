import numpy as np
from utils import softmax

class Env():
  def __init__(
    self,
    n_s: int,
    n_a: int,
    discount: float,
    init_dist: np.ndarray,
    transition_dist: np.ndarray,
  ):
    self.n_s = n_s
    self.states = np.arange(n_s)
    self.n_a = n_a
    self.actions = np.arange(n_a)
    self.discount = discount
    self.init_dist = init_dist
    self.transition_dist = transition_dist

class RandomEnv(Env):
  def __init__(self, n_s: int = 128, n_a: int = 16, discount: int = 0.9):
    init_dist = softmax(np.random.randn(n_s))
    transition_dist = softmax(np.random.randn(n_s, n_a, n_s))
    # IDEA: sample Gaussian, let the N highest values remain, 0 out the rest, then softmax
    super().__init__(n_s, n_a, discount, init_dist, transition_dist)
