from typing import Union
import torch
from env import Env
from distance.norm import norm
from _types import Reward

#! Does not converge for norm_ord inf
def minimal_potential_canon(
    reward: Reward, env: Env, norm_ord: Union[int, float], max_iters=1e6,
) -> Reward:
  r = torch.tensor(reward)
  # potential = torch.tensor(reward.mean(axis=(1, 2)), requires_grad=True)
  potential = torch.zeros(env.n_s, requires_grad=True)
  frozen_potential = torch.clone(potential)
  loss_frozen = float('inf')
 
  optimizer = torch.optim.Adam([potential], lr=1e-2)
  for i in range(int(max_iters)):
    optimizer.zero_grad()
    r_prime = r + env.discount * potential[None, None, :] - potential[:, None, None]
    loss = norm(r_prime, env, norm_ord)
    loss.backward()
    optimizer.step()
    # convergence = small gradient, potential hasn't changed in a while, or loss flattens
    if torch.norm(potential.grad, 2) < 1e-4: break
    if i%10000 == 0 and i != 0:
      if torch.isclose(potential, frozen_potential, rtol=1e-3, atol=1e-2).all():
        # print(i)
        break
      else: frozen_potential = torch.clone(potential)
    if i%1000 == 0:
      loss_val = loss.item()
      loss_is_close = abs(loss_val - loss_frozen) < 1e-8
      if loss_is_close: break
      loss_frozen = loss_val
    if i==max_iters-1:
      print("minimal_canon Didn't converge")
      return None
  return r_prime.detach().numpy()
