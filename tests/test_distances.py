import numpy as np
from copy import deepcopy
from distance import RewardDistance, canon
from coverage_dist import get_state_dist, get_action_dist
from tests.toy_env import env, reward

# D_s and D_a are arbitrary choices so we'll just go with the ones we defined
# in distance.canon
state_dist = get_state_dist(env)
action_dist = get_action_dist(env)

def test_state_dist():
  assert state_dist.sum() == 1

def test_action_dist():
  assert action_dist.sum() == 1

def test_epic_canon():
  expected_shaped = deepcopy(reward)

  # explicitly compute the expected values from the EPIC paper
  # (this is really slow for big envs which is the real EPIC function we use
  # is a lot more efficient, but this is easier to verify so good for testing)
  for s, s_vals in enumerate(reward):
    for a, a_vals in enumerate(s_vals):
      for s_prime, _ in enumerate(a_vals):

        expectation = 0
        for S, S_prob in enumerate(state_dist):
          for A, A_prob in enumerate(action_dist):
            for S_prime, S_prime_prob in enumerate(state_dist):
              prob = S_prob * A_prob * S_prime_prob
              expectation += prob * (
                env.discount * reward[s_prime, A, S_prime] -
                reward[s, A, S_prime] -
                env.discount * reward[S, A, S_prime]
              )

        expected_shaped[s, a, s_prime] += expectation

  output = canon.epic_canon(reward, env)
  assert np.isclose(output, expected_shaped).all()

def test_dard_canon():
  expected_shaped = deepcopy(reward)

  for s, s_vals in enumerate(reward):
    for a, a_vals in enumerate(s_vals):
      for s_prime, _ in enumerate(a_vals):

        expectation = 0
        for A, A_prob in enumerate(action_dist):
          for S_prime, S_prime_prob in enumerate(env.transition_dist[s, A]):
            for S_double, S_double_prob in enumerate(env.transition_dist[s_prime, A]):
              prob = A_prob * S_prime_prob * S_double_prob
              expectation += prob * (
                env.discount * reward[s_prime, A, S_double] -
                reward[s, A, S_prime] -
                env.discount * reward[S_prime, A, S_double]
              )

        expected_shaped[s, a, s_prime] += expectation

  output = canon.dard_canon(reward, env)
  assert np.isclose(output, expected_shaped).all()

# TODO test the RewardDistance class
