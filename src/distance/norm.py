import numpy as np
import torch
from typing import Union
from env import Env
from _types import Reward

def norm(reward: Reward, env: Env, ord: Union[int, float, str]) -> float:
  """Wrapper for np.linalg.norm that allows for weighted norms and baseline (no norm)
  ord: 0, 1, 2, 'weighted_1', 'weighted_2', 'weighted_inf', 'inf'
  """
  if ord == 0: return 1 # baseline (no norm)

  is_tensor = type(reward) is torch.Tensor

  if 'weighted' in str(ord): # weighted norms
    ord_num = float(ord.split('_')[1])
    if is_tensor: # torch code
      trans_dist = torch.tensor(env.transition_dist)
      if ord_num == np.inf:
          return torch.max(torch.abs(reward) * trans_dist)
      r = torch.abs(reward)
      r **= ord_num
      r *= trans_dist
      accum = torch.sum(r)
      accum **= 1 / ord_num
      return accum
      
    # numpy code
    if ord_num == np.inf:
        return np.max(np.abs(reward) * env.transition_dist)
    r = np.abs(reward)
    r **= ord_num
    r *= env.transition_dist
    accum = np.sum(r)
    accum **= 1 / ord_num
    return accum

  # regular norm
  if is_tensor:
    return torch.norm(reward, ord)
  return np.linalg.norm(reward.flatten(), ord)
