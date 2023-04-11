# taken from the EPIC paper section 5
import numpy as np
from env import Env
from _types import Reward

def init_epic_gridworlds(slippery: bool = False) -> tuple[Env, dict[str, Reward]]:
  n_s = 9 # 3x3 grid, 0 is top left, 8 is bottom right
  n_a = 5 # right, down, left, up, stay
  discount = 0.99

  init_dist = np.array([1, 0, 0, 0, 0, 0, 0, 0, 0])
  transition_dist = np.zeros((n_s, n_a, n_s))
  if slippery: transition_dist += 0.1 / (n_s-1)
  # fill transition_dist with probability of 1 of going in the desired direction
  # and 0 for all other directions
  for s in range(n_s):
    for a in range(n_a):
      if a == 0: # right
        if s % 3 != 2:
          transition_dist[s, a, s+1] = 0.9 if slippery else 1
      elif a == 1: # down
        if s < 6:
          transition_dist[s, a, s+3] = 0.9 if slippery else 1
      elif a == 2: # left
        if s % 3 != 0:
          transition_dist[s, a, s-1] = 0.9 if slippery else 1
      elif a == 3: # up
        if s > 2:
          transition_dist[s, a, s-3] = 0.9 if slippery else 1
      else: # stay
        transition_dist[s, a, s] = 0.9 if slippery else 1
  # for all (s, a) pairs that are not valid, set the transition to the current state
  for s in range(n_s):
    for a in range(n_a):
      if np.sum(transition_dist[s, a, :]) < 1:
        transition_dist[s, a, s] = 0.9 if slippery else 1
  env = Env(n_s, n_a, discount, init_dist, transition_dist)

  sparse_reward = np.zeros((n_s, n_a, n_s))
  sparse_reward[8, :, :] = 1

  dense_reward = np.zeros((n_s, n_a, n_s))
  for s in range(n_s-1):
    # reward of 2 for going to the right or down
    if s+1 < n_s: dense_reward[s, :, s+1] = 2
    if s+3 < n_s: dense_reward[s, :, s+3] = 2
    # reward of -4 for going to the left or up
    if s-1 >= 0: dense_reward[s, :, s-1] = -4
    if s-3 >= 0: dense_reward[s, :, s-3] = -4
    # reward of -1 for staying in the same place
    dense_reward[s, :, s] = -1
  # reward of 3 for staying in the bottom right corner
  dense_reward[8, :, 8] = 3

  path_reward = np.zeros((n_s, n_a, n_s))
  # reward of -1 in squares 1, 2, 6, 7
  path_reward[1, :, :] = -1
  path_reward[2, :, :] = -1
  path_reward[6, :, :] = -1
  path_reward[7, :, :] = -1
  # reward of 4 in square 8
  path_reward[8, :, :] = 4

  cliff_reward = np.zeros((n_s, n_a, n_s))
  # reward of -1 in squares 1 and 2
  cliff_reward[1, :, :] = -1
  cliff_reward[2, :, :] = -1
  # reward of -4 in squares 6 and 7
  cliff_reward[6, :, :] = -4
  cliff_reward[7, :, :] = -4
  # reward of 4 in square 8
  cliff_reward[8, :, :] = 4

  rewards = {
    'sparse': sparse_reward,
    'dense': dense_reward,
    'path': path_reward,
    'cliff': cliff_reward,
  }

  return env, rewards
