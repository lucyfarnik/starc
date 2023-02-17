import numpy as np
from utils import softmax

def test_softmax():
  _in = np.array([1, 2, 3])
  out = softmax(_in)
  expected = np.array([0.09, 0.2447, 0.6652])
  assert np.isclose(out, expected, atol=1e-3).all()

