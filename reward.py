import numpy as np
from env import Env
from _types import Reward

def random_reward(env: Env) -> Reward:
  return np.random.randn(env.n_s, env.n_a, env.n_s)

def interpolate(r1: Reward, r2: Reward, num=64) -> np.ndarray:
  diff = (r2 - r1) / num
  result = []
  for step in range(1, num+1):
    result.append(r1 + diff * step)

  return result
