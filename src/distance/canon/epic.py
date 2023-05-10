import torch
from env import Env
from _types import Reward
from distance.coverage_dist import get_state_dist, get_action_dist

def epic_canon(reward: Reward, env: Env) -> Reward:
  D_s = get_state_dist(env)
  D_a = get_action_dist(env)
  if type(reward) is torch.Tensor:
    D_s, D_a = torch.tensor(D_s), torch.tensor(D_a)
  S = D_s[:, None, None]
  A = D_a[None, :, None]
  S_prime = D_s[None, None, :]

  potential = (reward * A * S_prime).sum(axis=(1, 2))

  term1 = env.discount * potential[None, None, :]
  term2 = potential[:, None, None]
  term3 = env.discount * (reward * S * A * S_prime).sum()

  return reward + term1 - term2 - term3
