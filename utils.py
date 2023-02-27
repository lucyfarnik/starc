import jax.numpy as jnp
from jax import jit

# softmax along last dimension
def softmax(arr: jnp.ndarray) -> jnp.ndarray:
  exp = jnp.exp(arr)
  norm = jnp.sum(exp, axis=-1)
  norm = jnp.reshape(norm, (*arr.shape[0:-1], 1))
  norm = jnp.repeat(norm, arr.shape[-1], axis=-1)
  return exp / norm
softmax = jit(softmax)
