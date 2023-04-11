# D_s and D_a in the EPIC and DARD papers
#? Is there a better setting for these?
# In theory the best way is to make this correspond to spread among all policies
import numpy as np
from env import Env

# TODO
# think of multiple different distributions, modify the rest of the code
# to let you swap these around

# one way to do this would be to make it proportional to how often you'll be
# in that state given you follow a policy that wants to visit that state as
# often as possible, minus prob if policy wants to avoid that state
# -> this might minimize the regret bound?
# if this turns out to be equivalent-ish then we prove we don't have to
# do this fiddling with this; if it's different then we prove that the distribution
# matters
def get_state_dist(env: Env): # prob of transitioning into state
  return np.ones(env.n_s) / env.n_s
  # logits = env.transition_dist.sum(axis=(0, 1))
  # sum = logits.sum()
  # return logits / sum

def get_action_dist(env: Env): # right now this is just uniform
  return np.ones(env.n_a) / env.n_a
