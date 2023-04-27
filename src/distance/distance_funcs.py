import numpy as np
import torch
from distance.canon import epic_canon, dard_canon
from env import Env
from _types import Reward

def norm_func(r: Reward) -> float:
  if isinstance(r, torch.Tensor):
    normalized = torch.norm(r.flatten(), 2)
  else:
    normalized = np.linalg.norm(r.flatten(), 2)

  return normalized

def epic(r1: Reward, r2: Reward, env: Env) -> float:
  r1_can = epic_canon(r1, env)
  r2_can = epic_canon(r2, env)

  r1_size = norm_func(r1_can.flatten())
  if r1_size == 0:
    r1_norm = r1_can
  else:
    r1_norm = r1_can / r1_size
  r2_size = norm_func(r2_can.flatten())
  if r2_size == 0:
    r2_norm = r2_can
  else:
    r2_norm = r2_can / r2_size

  return norm_func((r1_norm - r2_norm).flatten())

def dard(r1: Reward, r2: Reward, env: Env) -> float:
  r1_can = dard_canon(r1, env)
  r2_can = dard_canon(r2, env)

  r1_norm = r1_can / norm_func(r1_can.flatten())
  r2_norm = r2_can / norm_func(r2_can.flatten())

  return norm_func((r1_norm - r2_norm).flatten())
