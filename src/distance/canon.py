import numpy as np
from typing import Union
import torch
from einops import rearrange
from env import Env
from _types import Reward
from distance.coverage_dist import get_state_dist, get_action_dist
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

#! Does not converge for norm_ord inf
# @timed
def minimal_canon(
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
    loss = torch.norm(r_prime, norm_ord)
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
      print("Didn't converge")
      return None #! FIXME
  return r_prime.detach().numpy()

# Take an arbitrary policy \pi (which could be the completely uniform policy, for example).
# Compute V^\pi.
# Let C(R)(s,a,s') = E_{S' ~ \tau(s,a)}[R(s,a,S') + gamma*V^\pi(S') - V^\pi(s)].
def state_val_exp_canon(reward: Reward, env: Env) -> Reward:
  # uniform probabilistic policy
  policy = np.ones((env.n_s, env.n_a)) / env.n_a
  
  # compute state values
  state_vals = np.zeros(env.n_s)
  for i in range(10000):
    prev_vals = state_vals.copy()
    for s in range(env.n_s):
      r_given_a_s_prime = reward[s] + env.discount * state_vals[None, :]
      r_dist = env.transition_dist[s] * policy[s, :, None] * r_given_a_s_prime
      state_vals[s] = r_dist.sum()
    if np.abs(state_vals - prev_vals).max() < 1e-8: break
    if i==9999: print("state_val_canon Didn't converge")
  
  # compute canonical reward
  canon = np.zeros_like(reward)
  for s in range(env.n_s):
    for a in range(env.n_a):
      r_given_s_prime = reward[s, a] + env.discount * state_vals - state_vals[s]
      canon[s, a, :] = (env.transition_dist[s, a] * r_given_s_prime).sum()
  
  return canon

# C(R)(s,a,s') = r(s,a,s') - V^pi(s) + gamma*V^pi(s')
def state_val_canon(reward: Reward, env: Env) -> Reward:
  policy = np.ones((env.n_s, env.n_a)) / env.n_a

  # compute state values
  state_vals = np.zeros(env.n_s)
  for i in range(10000):
    prev_vals = state_vals.copy()
    for s in range(env.n_s):
      r_given_a_s_prime = reward[s] + env.discount * state_vals[None, :]
      r_dist = env.transition_dist[s] * policy[s, :, None] * r_given_a_s_prime
      state_vals[s] = r_dist.sum()
    if np.abs(state_vals - prev_vals).max() < 1e-8: break
    if i==9999: print("state_val_canon Didn't converge")

  # compute canonical reward
  canon = np.zeros_like(reward)
  for s in range(env.n_s):
    for a in range(env.n_a):
      for s_prime in range(env.n_s):
        canon[s,a,s_prime] = reward[s,a,s_prime] - state_vals[s] + env.discount*state_vals[s_prime]
  
  return canon


canon_funcs = {
  'None': lambda r, _: r,
  'EPIC': epic_canon,
  'DARD': dard_canon,
  'Minimal': minimal_canon,
  'StateVal': state_val_canon,
  'StateValExp': state_val_exp_canon,
}

def norm_wrapper(reward: Reward, env: Env, ord: Union[int, float, str]) -> float:
  """Wrapper for np.linalg.norm that allows for weighted norms and baseline (no norm)
  ord: 0, 1, 2, 'weighted_1', 'weighted_2', 'weighted_inf', 'inf'
  """
  if ord == 0: return 1 # baseline (no norm)
  if 'weighted' in str(ord): # weighted norms
    ord_num = float(ord.split('_')[1])
    if ord_num == np.inf:
        return np.max(np.abs(reward) * env.transition_dist)
    r = np.abs(reward)
    r **= ord_num
    r *= env.transition_dist
    accum = np.sum(r)
    accum **= 1 / ord_num
    return accum

  # regular norm
  return np.linalg.norm(reward.flatten(), ord)

norm_opts = [1, 2, float('inf'), 'weighted_1', 'weighted_2', 'weighted_inf', 0]
# returns a dictionary of all the possible canonicalizations and normalizations
def canon_and_norm(reward: Reward, env: Env) -> dict[str, Reward]:
  can = {c_name: canon_funcs[c_name](reward, env)
         for c_name in ['None', 'EPIC', 'DARD', 'StateVal', 'StateValExp']}
  norm = {f'{c_name}-{n_ord}': val / norm_wrapper(val, env, n_ord)
            for n_ord in norm_opts
            for c_name, val in can.items()}
  # add in minimal canon (which depends on the norm order so it needs different code)
  for n_ord in norm_opts:
    if n_ord not in [1, 2]: continue #! REMOVE ME
    if n_ord == 0: continue
    min_can = canon_funcs['Minimal'](reward, env, n_ord)
    if min_can is None: continue # in case the canonicalization didn't converge
    norm[f'Minimal-{n_ord}'] = min_can / norm_wrapper(min_can, env, n_ord)
  return norm
