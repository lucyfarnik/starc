import numpy as np
from env import RandomEnv

# not super comprehensive right now, just checks that the distribution shapes
# are correct and that all distribution sum to 1
def test_random_env():
  env = RandomEnv()
  assert env.init_dist.shape == (env.n_s,)
  assert abs(env.init_dist.sum() - 1) < 0.01
  assert env.transition_dist.shape == (env.n_s, env.n_a, env.n_s)
  assert np.isclose(env.transition_dist.sum(axis=-1), np.ones((env.n_s, env.n_a))).all()
