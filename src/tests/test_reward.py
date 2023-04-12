import numpy as np
from env.reward import random_reward, interpolate, potential_shaping
from env import RandomEnv
from distance.distance_funcs import epic

def test_sparse_reward():
  for _ in range(100):
    env = RandomEnv()
    reward = random_reward(env, sparse=True, potential_shaped=False)
    assert reward.shape == (env.n_s, env.n_a, env.n_s)
    _, counts = np.unique(reward, return_counts=True)
    assert reward.size - counts.max() < env.n_s/2

def test_dense_reward():
  for _ in range(100):
    env = RandomEnv()
    reward = random_reward(env, sparse=False, potential_shaped=False)
    assert reward.shape == (env.n_s, env.n_a, env.n_s)
    _, counts = np.unique(reward, return_counts=True)
    assert counts.max() < 5

def test_state_dependent_reward():
  for _ in range(100):
    env = RandomEnv()
    reward = random_reward(env, state_dependent=True, potential_shaped=False)
    assert reward.shape == (env.n_s, env.n_a, env.n_s)
    for s in range(env.n_s):
      assert (reward[:, :, s] == reward[0, 0, s]).all()

def test_reward_only_in_terminal():
  for _ in range(100):
    env = RandomEnv(episodic=True)
    reward = random_reward(env, sparse=False, reward_only_in_terminal=True,
                           potential_shaped=False)
    assert reward.shape == (env.n_s, env.n_a, env.n_s)
    _, counts = np.unique(reward, return_counts=True)
    assert reward.size - counts.max() < env.n_s * env.n_a

def test_potential_shaping():
  for _ in range(100):
    env = RandomEnv()
    pre_shaping = random_reward(env, sparse=False, potential_shaped=False)
    post_shaping = potential_shaping(env, pre_shaping)
    assert epic(pre_shaping, post_shaping, env) < 1e-10

def test_interpolate():
  r1 = np.array([
    [0, 0],
    [1, 1],
    [2, 2]
  ])
  r2 = r1 + 3
  out = interpolate(r1, r2, num=3)
  for i, r_i in enumerate(out):
    assert (r_i == r1 + i+1).all()
