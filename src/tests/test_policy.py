import numpy as np
from src.distance.rollout import optimize, policy_returns, policy_return
from tests.toy_env import env, reward, expected_policy, expected_q_vals

def test_optimize():
  output = optimize(env, reward)
  assert np.isclose(output, expected_policy, atol=0.1).all()

def test_policy_returns():
  expected_return = expected_q_vals[:, expected_policy].mean()
  output = policy_returns([reward], expected_policy, env)[0]
  assert abs(output - expected_return) < 0.25
  #? Sometimes we get answers that are off by 0.2, is that something to be worried about?

  # make sure if we pass multiple rewards, it outputs multiple numbers
  out1, out2 = policy_returns([reward, reward], expected_policy, env)
  assert abs(out1 - expected_return) < 0.25
  assert abs(out2 - expected_return) < 0.25

def test_policy_return_wrapper():
  expected_return = expected_q_vals[:, expected_policy].mean()
  output = policy_return(reward, expected_policy, env)
  assert abs(output - expected_return) < 0.25
