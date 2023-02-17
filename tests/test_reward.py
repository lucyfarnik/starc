import numpy as np
from reward import random_reward, interpolate
from env import RandomEnv

def test_random_reward():
  env = RandomEnv()
  reward = random_reward(env)
  assert reward.shape == (env.n_s, env.n_a, env.n_s)
  mean = reward.mean()
  assert abs(mean)< 0.01
  std = reward.std()
  assert abs(std - 1) < 0.01

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
