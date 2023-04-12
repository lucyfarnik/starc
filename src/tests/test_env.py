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

def test_episodic_env():
  env = RandomEnv(episodic=True)
  assert env.init_dist.shape == (env.n_s,)
  assert abs(env.init_dist.sum() - 1) < 0.01
  assert env.transition_dist.shape == (env.n_s, env.n_a, env.n_s)
  assert np.isclose(env.transition_dist.sum(axis=-1), np.ones((env.n_s, env.n_a))).all()
  # make sure there's a terminal (self-looping) state
  for a in range(env.n_a):
    assert env.transition_dist[env.terminal_state, a, env.terminal_state] == 1
