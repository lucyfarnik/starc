import numpy as np

# softmax along last dimension
def softmax(arr: np.ndarray) -> np.ndarray:
  exp = np.exp(arr)
  norm = np.sum(exp, axis=-1)
  norm = np.reshape(norm, (*arr.shape[0:-1], 1))
  norm = np.repeat(norm, arr.shape[-1], axis=-1)
  return exp / norm
