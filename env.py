import jax.numpy as jnp
from jax import random
from utils import softmax

class Env():
  def __init__(
    self,
    n_s: int,
    n_a: int,
    discount: float,
    init_dist: jnp.ndarray,
    transition_dist: jnp.ndarray,
  ):
    self.n_s = n_s
    self.states = jnp.arange(n_s)
    self.n_a = n_a
    self.actions = jnp.arange(n_a)
    self.discount = discount
    self.init_dist = init_dist
    self.transition_dist = transition_dist

class RandomEnv(Env):
  def __init__(self, subkeys: tuple[random.KeyArray], 
               n_s: int = 128, n_a: int = 16, discount: int = 0.9):
    # we get the distributions by sampling a Gaussian,
    # only keeping the largest values and setting the rest to 0,
    # and then softmaxing the result

    # TODO we currently pick the largest values as being 1.8sigma above mean
    # this means that if we have larger n_s we'll be picking more values
    # at this stage which will make transitions more uniform again
    init_dist = random.normal(subkeys[0], (n_s,))
    init_dist = jnp.where(init_dist > 1.8,
                         init_dist, jnp.zeros_like(init_dist)-20)
    init_dist = softmax(init_dist)

    transition_dist = random.normal(subkeys[1], (n_s, n_a, n_s))
    transition_dist = jnp.where(transition_dist > 1.8,
                         transition_dist, jnp.zeros_like(transition_dist)-20)
    transition_dist = softmax(transition_dist)

    super().__init__(n_s, n_a, discount, init_dist, transition_dist)
