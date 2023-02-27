import jax.numpy as jnp
from utils import softmax

def test_softmax():
  _in = jnp.array([1, 2, 3])
  out = softmax(_in)
  expected = jnp.array([0.09, 0.2447, 0.6652])
  assert jnp.isclose(out, expected, atol=1e-3).all()

