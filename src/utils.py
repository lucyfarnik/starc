import numpy as np
from functools import wraps, partial
import time
from _types import Space, RewardCont

# softmax along last dimension
def softmax(arr: np.ndarray) -> np.ndarray:
  exp = np.exp(arr)
  norm = np.sum(exp, axis=-1)
  norm = np.reshape(norm, (*arr.shape[0:-1], 1))
  norm = np.repeat(norm, arr.shape[-1], axis=-1)
  return exp / norm

def timed(f):
  @wraps(f)
  def wrapped(*args, **kwargs):
    st = time.perf_counter()
    out = f(*args, **kwargs)
    et = time.perf_counter()
    print(f'{f.__name__} took {et-st:.4f}s')
    return out
  return wrapped

# sampling a state/action space (ie sampling a list of intervals)
def sample_space(space: Space) -> float:
  return [np.random.uniform(*interval) for interval in space]

def _reward_diff(r1: RewardCont, r2: RewardCont, *args):
  return r1(*args) - r2(*args)

def get_reward_diff(r1: RewardCont, r2: RewardCont):
  return partial(_reward_diff, r1, r2)
