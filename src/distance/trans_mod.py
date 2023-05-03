# implements d_tau and d_omega

import numpy as np
import torch
from typing import Callable
from env import Env
from _types import Reward
from utils import timed

# R^\tau for minimal canonicalization function and L2 norm
# For each state s and action a, pick a state s_{sa} such that τ(s, a, s_{sa}) > 0.
# Create a variable φ_s for every state s.
# Create a variable r_{sas'} for every transition s, a, s', where s' ≠ s_{sa}.
# Let R_2(s, a, s') = r_{sas'} for every transition s, a, s', where s' ≠ s_{sa}.
# Let R_2(s, a, s_{sa}) = (1/τ(s, a, s_{sa})) * ( Σ_{s' ∈ S} τ(s, a, s') * (R_1(s, a, s') + γ * φ_s') - φ_s - Σ_{s' ≠ s_{sa}} τ(s,a,s') * r_{sas'} )
# Then do gradient descent on all the variables with the objective to minimise the L_2-norm of R_2.
@timed
def r_tau_min(reward: Reward, env: Env, max_iters=1e6) -> Reward:
  # convert to torch
  if type(reward) is not torch.Tensor:
    reward = torch.tensor(reward)
  trans_dist = torch.tensor(env.transition_dist)

  # determine default action
  shaped_s = trans_dist.argmax(axis=-1)
  phi = torch.zeros(env.n_s, requires_grad=True)
  r = torch.zeros((env.n_s, env.n_a, env.n_s), requires_grad=True)
  optimizer = torch.optim.Adam([phi, r])

  r_frozen = torch.clone(r)
  phi_frozen = torch.clone(phi)
  loss_frozen = float('inf')

  for i in range(int(max_iters)):
    optimizer.zero_grad()
    r_2 = r.clone()

    # tau
    prob = torch.zeros((env.n_s, env.n_a))
    for s in range(env.n_s):
      for a in range(env.n_a):
        prob[s, a] = trans_dist[s, a, shaped_s[s, a]] # [s, a]

    # r1 shaped
    r1_add_potential = reward + env.discount * phi[None, None, :] # [s, a, s']
    r1_add_summed = (trans_dist * r1_add_potential).sum(dim=-1) # [s, a]
    r1_shaped = r1_add_summed - phi[:, None] # [s, a]

    # r2 shaped for the other s' values
    r2_all = (trans_dist * r).sum(dim=-1) # [s, a]
    r2_default = torch.zeros((env.n_s, env.n_a))
    for s in range(env.n_s):
      for a in range(env.n_a):
        r2_default[s, a] = prob[s, a] * r[s, a, shaped_s[s, a]] # [s, a]
    r2_others = r2_all - r2_default # [s, a]

    for s in range(env.n_s):
      for a in range(env.n_a):
        r_2[s, a, shaped_s[s, a]] = 1/prob[s, a] * r1_shaped[s, a] - r2_others[s, a]

    loss = torch.norm(r_2, 2)
    loss.backward()
    optimizer.step()

    # convergence checks
    if i%1000 == 0: # loss change over 1k iters is under 1e-8
      loss_val = loss.item()
      print(f'{loss_val=:<24} delta={loss_frozen - loss_val}')
      loss_is_close = abs(loss_val - loss_frozen) < 1e-8
      if loss_is_close: break
      loss_frozen = loss_val
    if i%10000 == 0 and i > 0: # param change over 10k iters is under 1e-2
      r_is_close = torch.isclose(r, r_frozen, atol=1e-2).all()
      phi_is_close = torch.isclose(phi, phi_frozen, atol=1e-2).all()
      if (r_is_close and phi_is_close):
        break
      else:
        print(f"iter={i:<10,} loss={loss.item():<12.3f} delta r={(r - r_frozen).abs().max():<10.3f} delta phi={(phi - phi_frozen).abs().max():.3f}")
        r_frozen = torch.clone(r)
        phi_frozen = torch.clone(phi)
    if loss < 1e-5: # extremely small loss
      break
    if i==max_iters-1: # didn't converge
      print("Didn't converge")
      return None #! FIXME

  return r_2.detach().numpy()

DistanceFunc = Callable[[Reward, Reward, Env], float]
@timed
def d_tau_min(r1: Reward, r2: Reward, env: Env, d_func: DistanceFunc) -> float:
  r1_tau = r_tau_min(r1, env)
  r2_tau = r_tau_min(r2, env)
  return d_func(r1_tau, r2_tau, env)
