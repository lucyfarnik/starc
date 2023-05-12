import numpy as np
from env import Env
from _types import Reward

# Take an arbitrary policy \pi (which could be the completely uniform policy, for example).
# Compute V^\pi.
# Let C(R)(s,a,s') = E_{S' ~ \tau(s,a)}[R(s,a,S') + gamma*V^\pi(S') - V^\pi(s)].
def value_canon(reward: Reward, env: Env) -> Reward:
  # uniform probabilistic policy
  policy = np.ones((env.n_s, env.n_a)) / env.n_a
  
  # compute state values
  state_vals = np.zeros(env.n_s)
  for i in range(10000):
    prev_vals = state_vals.copy()
    for s in range(env.n_s):
      r_given_a_s_prime = reward[s] + env.discount * state_vals[None, :]
      r_dist = env.transition_dist[s] * policy[s, :, None] * r_given_a_s_prime
      state_vals[s] = r_dist.sum()
    if np.abs(state_vals - prev_vals).max() < 1e-8: break
    if i==9999: print("state_val_canon Didn't converge")
  
  # compute canonical reward
  canon = np.zeros_like(reward)
  for s in range(env.n_s):
    for a in range(env.n_a):
      r_given_s_prime = reward[s, a] + env.discount * state_vals - state_vals[s]
      canon[s, a, :] = (env.transition_dist[s, a] * r_given_s_prime).sum()
  
  return canon
