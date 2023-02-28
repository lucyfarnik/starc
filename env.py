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
    # we get the distributions by sampling a Gaussian,
    # only keeping the largest values and setting the rest to 0,
    # and then softmaxing the result

    # TODO we currently pick the largest values as being 1.8sigma above mean
    # this means that if we have larger n_s we'll be picking more values
    # at this stage which will make transitions more uniform again
    thresh = 1 if n_s < 50 else (1.5 if n_s < 100 else 1.8) #! bigly goodn't hacky shit, kinda works for 32, 64, and 128
    init_dist = np.random.randn(n_s)
    init_dist = np.where(init_dist > thresh,
                         init_dist, np.zeros_like(init_dist)-20)
    init_dist = softmax(init_dist)

    transition_dist = np.random.randn(n_s, n_a, n_s)
    transition_dist = np.where(transition_dist > thresh,
                         transition_dist, np.zeros_like(transition_dist)-20)
    transition_dist = softmax(transition_dist)

    super().__init__(n_s, n_a, discount, init_dist, transition_dist)
