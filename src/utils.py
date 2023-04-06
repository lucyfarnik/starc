import numpy as np
from functools import wraps
import time

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
