# D_s and D_a in the EPIC and DARD papers
#? Is there a better setting for these?
# In theory the best way is to make this correspond to spread among all policies
import numpy as np
from env import Env

# TODO
# think of multiple different distributions, modify the rest of the code
# to let you swap these around

def get_state_dist(env: Env): # prob of transitioning into state
  #! TODO Just make it uniform so that it's easiest to compare stuff
  logits = env.transition_dist.sum(axis=(0, 1))
  sum = logits.sum()
  return logits / sum

def get_action_dist(env: Env): # right now this is just uniform
  return np.ones_like(env.actions) / env.n_a
