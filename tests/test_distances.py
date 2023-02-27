import numpy as np
from copy import deepcopy
from distance import RewardDistance, canon
from env import Env, RandomEnv
from coverage_dist import get_state_dist, get_action_dist
from reward import random_reward
from _types import Reward
from tests.toy_env import env, reward

# D_s and D_a are arbitrary choices so we'll just go with the ones we defined
# in distance.canon
state_dist = get_state_dist(env)
action_dist = get_action_dist(env)

def test_state_dist():
  assert state_dist.sum() == 1

def test_action_dist():
  assert action_dist.sum() == 1

# explicitly compute the expected values from the EPIC paper
# (this is really slow for big envs which is the real EPIC function we use
# is a lot more efficient, but this is easier to verify so good for testing)
def slow_epic(r: Reward, e: Env):
  state_dist = get_state_dist(e)
  action_dist = get_action_dist(e)

  term1 = np.zeros((1, 1, e.n_s))
  for s_prime in range(e.n_s):
    for A, A_prob in enumerate(action_dist):
      for S_prime, S_prime_prob in enumerate(state_dist):
        prob = A_prob * S_prime_prob
        term1[0, 0, s_prime] += prob * (
          e.discount * r[s_prime, A, S_prime]
        ) 

  term2 = np.zeros((e.n_s, 1, 1))
  for s in range(e.n_s):
    for A, A_prob in enumerate(action_dist):
      for S_prime, S_prime_prob in enumerate(state_dist):
        prob = A_prob * S_prime_prob
        term2[s, 0, 0] += prob * r[s, A, S_prime]

  term3 = 0
  for S, S_prob in enumerate(state_dist):
    for A, A_prob in enumerate(action_dist):
      for S_prime, S_prime_prob in enumerate(state_dist):
        prob = S_prob * A_prob * S_prime_prob
        term3 += prob * e.discount * r[S, A, S_prime]

  return r + term1 - term2 - term3

def test_epic_canon_toy():
  expected_shaped = slow_epic(reward, env)
  output = canon.epic_canon(reward, env)
  assert np.isclose(output, expected_shaped).all()

def test_epic_canon():
  e = RandomEnv(n_s=16, n_a=4)
  r = random_reward(e)
  expected = slow_epic(r, e)
  output = canon.epic_canon(r, e)
  assert np.isclose(output, expected).all()

def slow_dard(r: Reward, e: Env):
  action_dist = get_action_dist(e)

  term1 = np.zeros((1, 1, e.n_s))
  for s_prime in range(e.n_s):
    for A, A_prob in enumerate(action_dist):
      for S_double, S_double_prob in enumerate(e.transition_dist[s_prime, A, :]):
        prob = A_prob * S_double_prob
        term1[0, 0, s_prime] += prob * e.discount * r[s_prime, A, S_double]
  
  term2 = np.zeros((e.n_s, 1, 1))
  for s in range(e.n_s):
    for A, A_prob in enumerate(action_dist):
      for S_prime, S_prime_prob in enumerate(e.transition_dist[s, A, :]):
        prob = A_prob * S_prime_prob
        term2[s, 0, 0] += prob * r[s, A, S_prime]
  
  term3 = np.zeros((e.n_s, 1, e.n_s))
  for s in range(e.n_s):
    for s_prime in range(e.n_s):
      for A, A_prob in enumerate(action_dist):
        for S_prime, S_prime_prob in enumerate(e.transition_dist[s, A, :]):
          for S_double, S_double_prob in enumerate(e.transition_dist[s_prime, A, :]):
            prob = A_prob * S_prime_prob * S_double_prob
            term3[s, 0, s_prime] += prob * e.discount * r[S_prime, A, S_double]
  
  return r + term1 - term2 - term3

def test_dard_canon_toy():
  expected_shaped = slow_dard(reward, env) 
  output = canon.dard_canon(reward, env)
  assert np.isclose(output, expected_shaped).all()

def test_dard_canon():
  e = RandomEnv(n_s=16, n_a=4)
  r = random_reward(e)
  expected = slow_dard(r, e)
  output = canon.dard_canon(r, e)
  assert np.isclose(output, expected).all()

# this is just a sanity check to make sure the numpy function does what
# I think it does
def test_norms():
  test_in = np.array([1, 2, 3])
  assert np.linalg.norm(test_in, 1) == 6
  assert np.linalg.norm(test_in, 2) - 3.741657 < 1e-5
  assert np.linalg.norm(test_in, float('inf')) == 3

def test_distance():
  e = RandomEnv(n_s=16, n_a=4)
  r1, r2 = random_reward(e), random_reward(e)
  exp_can1, exp_can2 = slow_epic(r1, e), slow_epic(r2, e)
  stand1 = exp_can1 / np.linalg.norm(exp_can1.flatten(), 2)
  stand2 = exp_can2 / np.linalg.norm(exp_can2.flatten(), 2)
  exp = np.linalg.norm((stand1 - stand2).flatten(), 2)
  out = RewardDistance('EPIC', 2, 2)(r1, r2, e)
  assert np.isclose(exp, out).all()
