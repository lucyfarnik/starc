import jax.numpy as jnp

# just here for readability in the other files
Reward = jnp.ndarray # [S, A, S']; reward at transition s -a-> s'
Policy = jnp.ndarray # [S] action to take in state; deterministic
