import numpy as np
from distance.canon import epic_canon, dard_canon, minimal_potential_canon, state_val_canon
from distance.canon import canon_and_norm
from distance.norm import norm
from env import Env, RandomEnv
from distance.coverage_dist import get_state_dist, get_action_dist
from env.reward import random_reward, potential_shaping
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
  output = epic_canon(reward, env)
  assert np.isclose(output, expected_shaped).all()

def test_epic_canon():
  e = RandomEnv(n_s=16, n_a=4)
  r = random_reward(e)
  expected = slow_epic(r, e)
  output = epic_canon(r, e)
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
  output = dard_canon(reward, env)
  assert np.isclose(output, expected_shaped).all()

def test_dard_canon():
  e = RandomEnv(n_s=16, n_a=4)
  r = random_reward(e)
  expected = slow_dard(r, e)
  output = dard_canon(r, e)
  assert np.isclose(output, expected).all()

def test_minimal_canon_removes_potential_shaping():
  for _ in range(10):
    e = RandomEnv(n_s=16, n_a=4)
    r = np.zeros((e.n_s, e.n_a, e.n_s))
    shaped_r = potential_shaping(e, r)
    output = minimal_potential_canon(shaped_r, e, 2)
    assert np.isclose(output, r, atol=1e-2).all()

def slow_minimal(r: Reward, e: Env):
  assert False, 'TODO implement this'

def test_minimal_canon_toy():
  expected_shaped = slow_minimal(reward, env) 
  output = minimal_potential_canon(reward, env)
  assert np.isclose(output, expected_shaped).all()

def test_minimal_canon():
  e = RandomEnv(n_s=16, n_a=4)
  r = random_reward(e)
  expected = slow_minimal(r, e)
  output = minimal_potential_canon(r, e)
  assert np.isclose(output, expected).all()

def test_state_val_canon():
  for _ in range(10):
    e = RandomEnv(n_s=16, n_a=4)
    r1 = random_reward(e)
    r2 = potential_shaping(e, r1)
    assert np.isclose(state_val_canon(r1, e), state_val_canon(r2, e), atol=1e-6).all()

# this is just a sanity check to make sure the numpy function does what
# I think it does
def test_norms():
  test_in = np.array([1, 2, 3])
  assert np.linalg.norm(test_in, 1) == 6
  assert np.linalg.norm(test_in, 2) - 3.741657 < 1e-5
  assert np.linalg.norm(test_in, float('inf')) == 3

def test_norm():
  test_in = np.array([1, 2, 3])
  assert norm(test_in, None, 1) == 6
  assert norm(test_in, None, 0) == 1

  env = Env(n_s=2, n_a=2, discount=0.9, init_dist=np.array([0.5, 0.5]),
            transition_dist=np.array([[[0.8, 0.2], [0.8, 0.2]],
                                      [[0.8, 0.2], [0.8, 0.2]]]))
  r = np.array([[[1.0, 2.0], [3.0, 4.0]], [[5.0, 6.0], [7.0, 8.0]]])
  assert abs(norm(r, env, 'weighted_1') - 16.8) < 1e-5

def test_canon_and_norm():
  e = RandomEnv(n_s=32, n_a=4)
  r = random_reward(e)
  r_epic = slow_epic(r, e)
  r_dard = slow_dard(r, e)
  r_minimal = slow_minimal(r, e)

  expected = {}
  expected['None-0'] = r
  expected['EPIC-0'] = r_epic
  expected['DARD-0'] = r_dard
  expected['Minimal-0'] = r_minimal
  expected['StateVal-0'] = None
  for n in [1, 2, float('inf')]:
    expected[f'None-{n}'] = r / np.linalg.norm(r.flatten(), n)
    expected[f'EPIC-{n}'] = r_epic / np.linalg.norm(r_epic.flatten(), n)
    expected[f'DARD-{n}'] = r_dard / np.linalg.norm(r_dard.flatten(), n)
    expected[f'Minimal-{n}'] = r_minimal / np.linalg.norm(r_minimal.flatten(), n)
    expected[f'StateVal-{n}'] = None

  output = canon_and_norm(r, e)

  assert expected.keys() == output.keys()
  for key, val in expected.items():
    if 'StateVal' in key: continue
    assert np.isclose(val, output[key]).all()
    