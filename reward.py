import numpy as np
from typing import Optional
from env import Env
from _types import Reward

# maybe also have different Gaussian means for different states? move all up or down etc
# sparse can be a boolean, or None which means we randomly decide
def random_reward(env: Env,
                  sparse: Optional[bool] = None,
                  state_dependent: bool = False,
                  reward_only_in_terminal: bool = False) -> Reward:
  """
    Create a random reward function.
    env: the environment
    sparse: whether the reward function should be sparse (ie all but around 3 are zero)
    state_dependent: whether the reward should depend exclusive on the state
      (rather than the transition)
    reward_only_in_terminal: whether the reward should only be nonzero when
      transitioning into the terminal state
  """
  if reward_only_in_terminal:
    assert hasattr(env, 'terminal_state'), 'env must be episodic for reward_only_in_terminal'
    r = np.zeros((env.n_s, env.n_a, env.n_s))
    for s in range(env.n_s):
      if s != env.terminal_state:
        r[s, :, env.terminal_state] = 1 + np.random.randn(env.n_a)
    return r
  
  if state_dependent:
    state_r = np.random.randn(env.n_s)
    r = np.zeros((env.n_s, env.n_a, env.n_s))
    for s in range(env.n_s):
      r[:, :, s] = state_r[s]
  else:
    r = np.random.randn(env.n_s, env.n_a, env.n_s) # iid Gaussian
  if sparse is True or (sparse is None and np.random.random() > 0.8): # make it sparse sometimes
    # determine the threshold to use for sparseness
    #! bigly goodn't hacky shit, kinda works for 32, 64, and 128
    thresh = 3 if env.n_s < 50 else (3.5 if env.n_s < 100 else 3.8)
    if state_dependent: thresh -= 2 # if it's state dependent then there's only n_s unique vals
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
