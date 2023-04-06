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
  def __init__(self, n_s: int = 128, n_a: int = 16, discount: int = 0.9,
               episodic: bool = False):
    """
      Create a random environment.
      n_s: number of states
      n_a: number of actions
      discount: discount factor
      episodic: whether the environment is episodic - in this case episodic
        means that there is a terminal (self-looping) state
    """
    # uniform init dist
    init_dist = np.ones(n_s) / n_s

    # sample iid Gaussians, then only keep the highest values, then softmax -> sparse
    # TODO we currently pick the largest values as being 1.8sigma above mean
    # this means that if we have larger n_s we'll be picking more values
    # at this stage which will make transitions more uniform again
    thresh = 1 if n_s < 50 else (1.5 if n_s < 100 else 1.8) #! bigly goodn't hacky shit, kinda works for 32, 64, and 128
    transition_dist = np.random.randn(n_s, n_a, n_s)
    transition_dist = np.where(transition_dist > thresh,
                         transition_dist, np.zeros_like(transition_dist)-20)
    transition_dist = softmax(transition_dist)

    self.episodic = episodic
    if episodic:
      self.terminal_state = np.random.randint(0, n_s)
      for a in range(n_a):
        for s_prime in range(n_s):
          prob = 1.0 if s_prime == self.terminal_state else 0.0
          transition_dist[self.terminal_state, a, s_prime] = prob

    super().__init__(n_s, n_a, discount, init_dist, transition_dist)
