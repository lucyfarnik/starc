import numpy as np
import torch
from einops import rearrange
from env import Env
from _types import Reward
from coverage_dist import get_state_dist, get_action_dist
from utils import timed

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

def dard_canon(reward: Reward, env: Env) -> Reward:
  A = get_action_dist(env)

  potential = (env.transition_dist * reward).sum(axis=2)
  potential = (potential * A[None, :]).sum(axis=1)

  term1 = env.discount * potential[None, None, :]
  term2 = potential[:, None, None]
  
  joint_probs = ( # [s, s', S', A, S'']; p(S', S'' | s, s', A=A)
    A[None, None, None, :, None] * 
    rearrange(env.transition_dist, 's A Sp -> s 1 Sp A 1') *
    rearrange(env.transition_dist, 'sp A Sd -> 1 sp 1 A Sd')
  )
  r_given_probs = reward[None, None, ...] * joint_probs
  term3 = env.discount * r_given_probs.sum(axis=(2,3,4))[:,None,:]
  
  return reward + term1 - term2 - term3

#! Does not converge for norm_ord 1 or inf
# @timed
def minimal_canon(reward: Reward, env: Env, norm_ord: int) -> Reward:
  r = torch.tensor(reward)
  potential = torch.zeros(env.n_s, requires_grad=True)
  optimizer = torch.optim.Adam([potential])
  for _ in range(20000):
    optimizer.zero_grad()
    r_prime = r + env.discount * potential[None, None, :] - potential[:, None, None]
    loss = torch.norm(r_prime, norm_ord)
    loss.backward()
    optimizer.step()
    if torch.norm(potential.grad, 2) < 1e-4: break
  return r_prime.detach().numpy()

canon_funcs = {
  'None': lambda r, _: r,
  'EPIC': epic_canon,
  'DARD': dard_canon,
  'Minimal': minimal_canon,
}

# computes either the norm, or returns 1 if ord==0
# which makes it useful in defining canon_and_norm (where norm==0 means don't normalize)
def norm_wrapper(reward: Reward, ord: int) -> float:
  if ord == 0: return 1
  return np.linalg.norm(reward.flatten(), ord)

norm_opts = [1, 2, float('inf'), 0]
# returns a dictionary of all the possible canonicalizations and normalizations
def canon_and_norm(reward: Reward, env: Env) -> dict[str, Reward]:
  can = {c_name: canon_funcs[c_name](reward, env)
         for c_name in ['None', 'EPIC', 'DARD']}
  norm = {f'{c_name}-{n_ord}': val / norm_wrapper(val, n_ord)
            for n_ord in norm_opts
            for c_name, val in can.items()}
  # add in minimal canon (which depends on the norm order so it needs different code)
  for n_ord in norm_opts:
    if n_ord == 0: continue
    min_can = canon_funcs['Minimal'](reward, env, n_ord)
    norm[f'Minimal-{n_ord}'] = min_can / norm_wrapper(min_can, n_ord)
  return norm
