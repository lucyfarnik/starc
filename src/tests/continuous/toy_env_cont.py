import numpy as np
from _types import EnvInfoCont

state_space = [(-10, 10), (-5, 5)]
action_space = [(-1, 1), (-1, 1)]
discount = 0.9
def trans_dist(s, a):
  s_prime = [
    s[0] + a[0] + np.random.normal(0, 0.1),
    s[1] + a[1] + np.random.normal(0, 0.1)
  ]

  # if we go beyond the bounds, bounce back
  for idx in range(len(s_prime)):
    if s_prime[idx] < state_space[idx][0]:
      s_prime[idx] = 2 * state_space[idx][0] - s_prime[idx]
    elif s_prime[idx] > state_space[idx][1]:
      s_prime[idx] = 2 * state_space[idx][1] - s_prime[idx]

  return s_prime

def trans_prob(s, a, s_prime):
  return 0.01

def reward(s, a, s_prime):
  return s[0] * (s_prime[0] - s[0]) + s[1] * (s[1] - s_prime[1]) + np.random.normal(0, 1)

def state_vals(s) -> float: # note: this isn't actually accurate, it assumes discount=0
  val = 0

  if s[0] > 9: val += s[0] * (10-s[0])
  else: val += s[0]

  if s[1] < -4: val += s[1] * (-4-s[1])
  else: val += s[1]

  return val

env_info = EnvInfoCont(trans_dist, trans_prob, discount, state_space, action_space, state_vals)
