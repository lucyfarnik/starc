# most of implementation taken from transition_graph.py that Joar sent over
import numpy as np
from env import Env
from _types import Reward
from coverage_dist import get_state_dist, get_action_dist


def gradient(potential: np.ndarray, env: Env) -> np.ndarray:
  return env.discount * potential[None, None, :] - potential[:, None, None]

def epic_canon(reward: Reward, env: Env) -> Reward:
  # add dummy dimensions to the state dist and action dist we're sampling from
  state_dist = get_state_dist(env)[None, None, :]
  action_dist = get_action_dist(env)[None, :, None]
  potential = (reward * action_dist * state_dist).sum(axis=(1, 2))
  avg_r = (potential * state_dist.flatten()).sum()
  return reward + gradient(potential, env) - env.discount * avg_r


#! DOESN'T WORK - FIXING IT IN ___dard_fuckery.ipynb
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