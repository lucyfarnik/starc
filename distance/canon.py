import numpy as np
from env import Env
from _types import Reward
from coverage_dist import get_state_dist, get_action_dist

def epic_canon(reward: Reward, env: Env) -> Reward:
  D_s = get_state_dist(env)
  D_a = get_action_dist(env)
  S = D_s[:, None, None]
  A = D_a[None, :, None]
  S_prime = D_s[None, None, :]

  potential = (reward * A * S_prime).sum(axis=(1, 2))

  term1 = env.discount * potential[None, None, :]
  term2 = potential[:, None, None]
  term3 = env.discount * (reward * S * A * S_prime).sum()

  return reward + term1 - term2 - term3


#! DOESN'T WORK - FIXING IT IN ___dard_fuckery.ipynb
# taken from the file Joar sent over
def gradient(potential: np.ndarray, env: Env) -> np.ndarray:
  return env.discount * potential[None, None, :] - potential[:, None, None]

def dard_canon(reward: Reward, env: Env) -> Reward:
  # weights = (
  #   env.state_dist[:, None, None] *
  #   env.action_dist[None, :, None] *
  #   env.state_dist[None, None, :]
  # )
  # weighted_r = reward * weights
  weighted_r = env.transition_dist * reward
  outflow = weighted_r.sum((-2, -1))
  potential = outflow / get_state_dist(env)
  joint_state_probs = env.transition_dist.sum(axis=1)
  next_state_probs = joint_state_probs / joint_state_probs.sum(
      axis=1, keepdims=True
  )
  action_probs = get_action_dist(env)
  offset = (
      reward[None, :, :, None, :]
      * next_state_probs[:, :, None, None, None]
      * action_probs[None, None, :, None, None]
      * next_state_probs[None, None, None, :, :]
  ).sum(axis=(1, 2, 4))

  return reward + gradient(potential, env) - env.discount * offset[:, None, :]


canon_funcs = {
  'EPIC': epic_canon,
  'DARD': dard_canon,
}

#! The divergence-free canon is prob gonna be solvable with torch.optim
# https://openreview.net/pdf?id=Hn21kZHiCK section 6 "minimal canonicalization c"