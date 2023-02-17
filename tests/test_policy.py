import numpy as np
from policy import optimize, policy_return
from tests.toy_env import env, reward, expected_policy, expected_q_vals

def test_optimize():
  output = optimize(env, reward)
  assert np.isclose(output, expected_policy, atol=0.1).all()

def test_policy_return():
  expected_return = expected_q_vals[:, expected_policy].mean()
  output = policy_return(reward, expected_policy, env)
  assert abs(output - expected_return) < 0.25
  #? Sometimes we get answers that are off by 0.2, is that something to be worried about?
  