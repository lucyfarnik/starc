import jax.numpy as jnp
from jax import random
from env import RandomEnv

# not super comprehensive right now, just checks that the distribution shapes
# are correct and that all distribution sum to 1
def test_random_env():
  key = random.PRNGKey(12345)
  subkeys = random.split(key)
  env = RandomEnv(subkeys)
  assert env.init_dist.shape == (env.n_s,)
  assert abs(env.init_dist.sum() - 1) < 0.01
  assert env.transition_dist.shape == (env.n_s, env.n_a, env.n_s)
  assert jnp.isclose(env.transition_dist.sum(axis=-1), jnp.ones((env.n_s, env.n_a))).all()
