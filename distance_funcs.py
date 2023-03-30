import numpy as np
import torch
from canon import epic_canon, dard_canon
from env import Env
from _types import Reward

def epic(r1: Reward, r2: Reward, env: Env) -> float:
  r1_can = epic_canon(r1, env)
  r2_can = epic_canon(r2, env)

  r1_norm = r1_can / np.linalg.norm(r1_can.flatten(), 2)
  r2_norm = r2_can / np.linalg.norm(r2_can.flatten(), 2)

  return np.linalg.norm((r1_norm - r2_norm).flatten(), 2)

def epic_torch(r1: Reward, r2: Reward, env: Env) -> float:
  r1_can = epic_canon(r1, env)
  r2_can = epic_canon(r2, env)

  r1_norm = r1_can / torch.norm(r1_can.flatten(), 2)
  r2_norm = r2_can / torch.norm(r2_can.flatten(), 2)

  return torch.norm((r1_norm - r2_norm).flatten(), 2)

def dard(r1: Reward, r2: Reward, env: Env) -> float:
  r1_can = dard_canon(r1, env)
  r2_can = dard_canon(r2, env)

  r1_norm = r1_can / np.linalg.norm(r1_can.flatten(), 2)
  r2_norm = r2_can / np.linalg.norm(r2_can.flatten(), 2)

  return np.linalg.norm((r1_norm - r2_norm).flatten(), 2)
