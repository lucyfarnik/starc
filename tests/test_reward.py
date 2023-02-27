import jax.numpy as jnp
from jax import random
from reward import random_reward, interpolate
from env import RandomEnv

def test_random_reward():
  rand_key = random.PRNGKey(12345)
  rand_key, *subkeys = random.split(rand_key, 3)
  env = RandomEnv(subkeys)
  reward = random_reward(env, rand_key)
  assert reward.shape == (env.n_s, env.n_a, env.n_s)
  mean = reward.mean()
  assert abs(mean)< 0.01
  std = reward.std()
  assert abs(std - 1) < 0.01

def test_interpolate():
  r1 = jnp.array([
    [0, 0],
    [1, 1],
    [2, 2]
  ])
  r2 = r1 + 3
  out = interpolate(r1, r2, num=3)
  for i, r_i in enumerate(out):
    assert (r_i == r1 + i+1).all()
