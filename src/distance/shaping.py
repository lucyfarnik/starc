import numpy as np
import torch
from typing import Callable
from env import Env
from _types import Reward

# R^\tau(s,a,s') = E_{S' ~ \tau(s,a)}[R(s,a,S')]
def transition_shaping(reward: Reward, env: Env) -> Reward:
  values = (env.transition_dist * reward).sum(axis=-1) # [s, a]
  # add dummy dimension for s'
  result = torch.zeros_like(reward) if type(reward) is torch.Tensor else np.empty(reward.shape)
  for i in range(result.shape[-1]):
    result[:,:,i] = values
  return result

# d^\tau(R1, R2) = d(R1^\tau, R2^\tau)
DistanceFunc = Callable[[Reward, Reward, Env], float]
def shaped_distance(r1: Reward, r2: Reward, env: Env, d_func: DistanceFunc) -> float:
  shaped1 = transition_shaping(r1, env)
  shaped2 = transition_shaping(r2, env)
  return d_func(shaped1, shaped2, env)

# d^\omega(R1, R2) = max_{\tau} d^\tau(R1, R2)
def maximal_transition_dist(r1: Reward, r2: Reward, env: Env, d_func: DistanceFunc,
                            max_iters: int = 100000) -> float:
  r1, r2 = torch.tensor(r1), torch.tensor(r2)
  trans_dist = torch.tensor(torch.randn(env.transition_dist.shape).softmax(dim=-1), requires_grad=True)
  optimizer = torch.optim.Adam([trans_dist])
  frozen_dist = 0 # another convergence criterion - if this stops changing, we're done
  for i in range(max_iters):
    optimizer.zero_grad()
    env_i = Env(n_s=env.n_s, n_a=env.n_a, discount=env.discount,
                init_dist=env.init_dist, transition_dist=trans_dist)
    dist = shaped_distance(r1, r2, env_i, d_func)
    loss = -1 * dist # we're maximizing, not minimizing here
    loss += (trans_dist.sum(dim=-1) - 1).abs().sum() # penalize invalid distributions
    loss.backward()
    optimizer.step()
    if torch.norm(trans_dist.grad, 2) < 1e-4:
      break
    if i % 100 == 0 and i > 0:
      if abs(frozen_dist - dist.item()) < 1e-5:
        break
      frozen_dist = dist.item()
    if i%10000 == 0 and i > 0:
      print(f"iter {i:,}, dist={dist}, invalid distribution penalty={(trans_dist.sum(dim=-1) - 1).abs().sum()}, loss {loss.item()}, grad size {torch.norm(trans_dist, 2)}\n\n")
  return dist.item(), trans_dist
