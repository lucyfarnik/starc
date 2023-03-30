import numpy as np
from typing import Optional
from env import Env
from _types import Reward

# maybe also have different Gaussian means for different states? move all up or down etc
def random_reward(env: Env, sparse: Optional[bool] = None) -> Reward:
  r = np.random.randn(env.n_s, env.n_a, env.n_s) # iid Gaussian
  if sparse is True or (sparse is None and np.random.random() > 0.8): # make it sparse sometimes
    thresh = 3 if env.n_s < 50 else (3.5 if env.n_s < 100 else 3.8) #! bigly goodn't hacky shit, kinda works for 32, 64, and 128
    r = np.where(r > thresh, r, np.zeros_like(r))
  if np.random.random() > 0.3: # scale it most of the time
    r *= 10 * np.random.random()
  if np.random.random() > 0.7: # move the whole thing up or down sometimes
    r += 10 * np.random.random()
  if np.random.random() > 0.5: # apply potential shaping half the time
    potential = np.random.randn(env.n_s)
    potential *= 10 * np.random.random() # scale
    potential += np.random.random() # move up or down
    r += env.discount * potential[None, None, :] - potential[:, None, None]
  return r

def sparse_reward(env: Type[Env]) -> Reward:
    r = np.random.randn(env.n_s, env.n_a, env.n_s)
    thresh = 3 if env.n_s < 50 else (3.5 if env.n_s < 100 else 3.8)
    r = np.where(r > thresh, r, np.zeros_like(r))
    r *= 10 * np.random.random()
    r += 10 * np.random.random()
    potential = np.random.randn(env.n_s)
    potential *= 10 * np.random.random()
    potential += np.random.random()
    r += env.discount * potential[None, None, :] - potential[:, None, None]
    return r

def dense_reward(env: Type[Env]) -> Reward:
    r = np.random.randn(env.n_s, env.n_a, env.n_s)
    r *= 10 * np.random.random()
    r += 10 * np.random.random()
    potential = np.random.randn(env.n_s)
    potential *= 10 * np.random.random()
    potential += np.random.random()
    r += env.discount * potential[None, None, :] - potential[:, None, None]
    return r


# return a list of rewards between r1 and r2
# the interpolation is logarithmic rather than linear (that way there's more
# steps closer to r1)
def interpolate(
    r1: Reward, r2: Reward,
    num=64,
    logarithmic=False,
) -> np.ndarray:
  if logarithmic: #! doesn't work rn, diff contains nans
    r1_no_zeros = np.where(r1 == 0, 1e-10*np.ones_like(r1), r1)
    diff = np.power(r2 / r1_no_zeros, 1/num)
    result = []
    for step in range(1, num+1):
      result.append(r1 * np.power(diff, step))

    return result
  else:
    diff = (r2 - r1) / num
    result = []
    for step in range(1, num+1):
      result.append(r1 + diff * step)

    return result
