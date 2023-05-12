from env import Env
from _types import Reward
from distance.canon.epic import epic_canon
from distance.canon.dard import dard_canon
from distance.canon.minimal_potential import minimal_potential_canon
from distance.canon.value_potential import value_potential_canon
from distance.canon.value import value_canon
from distance.canon.minimal import minimal_canon
from distance.norm import norm

canon_funcs = {
  'None': lambda r, _: r,
  'EPIC': epic_canon,
  'DARD': dard_canon,
  'MinimalPotential': minimal_potential_canon,
  'ValuePotential': value_potential_canon,
  'Value': value_canon,
  'Minimal': minimal_canon,
}

norm_opts = [1, 2, float('inf'), 'weighted_1', 'weighted_2', 'weighted_inf', 0]
# returns a dictionary of all the possible canonicalizations and normalizations
def canon_and_norm(reward: Reward, env: Env, incl_min_s_prime=True) -> dict[str, Reward]:
  can_r = {c_name: canon_funcs[c_name](reward, env)
         for c_name in canon_funcs.keys() if 'Minimal' not in c_name}
  norm_r = {f'{c_name}-{n_ord}': val / norm(val, env, n_ord)
            for n_ord in norm_opts
            for c_name, val in can_r.items()}
  # add in minimal_potential canon (which depends on the norm order so it needs different code)
  for n_ord in norm_opts:
    if 'inf' in str(n_ord) or n_ord == 0:
      continue
    min_can = canon_funcs['MinimalPotential'](reward, env, n_ord)
    if min_can is not None: # make sure it converged
      norm_r[f'MinimalPotential-{n_ord}'] = min_can / norm(min_can, env, n_ord) 

  # add minimal canon (which can return None if it doesn't converge)
  if incl_min_s_prime:
    min_s_prime = canon_funcs['Minimal'](reward, env)
    if min_s_prime is not None: # make sure it converged
      # we only do Minimal for norm_ord 2
      norm_r['Minimal-2'] = min_s_prime / norm(min_s_prime, env, 2)
  return norm_r
